from superpacman import make_env
import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torch.nn.functional import log_softmax
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from pathlib import Path
from time import time
from warnings import warn
from math import inf
from torchrl.record.loggers import get_logger, generate_exp_name
from torchrl.record import CSVLogger
from importlib.metadata import version, PackageNotFoundError
from hrid import HRID
import torch.cuda


class VGGConvBlock(nn.Module):
    CHANNEL_0 = 64
    CHANNEL_1 = 128
    def __init__(self, in_channels, batchnorm_momentum=0.001):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.CHANNEL_0, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.CHANNEL_0, momentum=batchnorm_momentum, track_running_stats=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.CHANNEL_0, out_channels=self.CHANNEL_1, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.CHANNEL_1, momentum=batchnorm_momentum, track_running_stats=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, image):
        if len(image.shape) == 5:
            N, L, C, H, W = image.shape
            image = image.flatten(0, 1)
            image = self.layers(image)
            image = image.unflatten(0, (N, L))
        else:
            image = self.layers(image)
        return image


class Value(nn.Module):
    """
    MLP value function
    """

    def __init__(self, in_features, in_channels, hidden_dim, batchnorm_momentum=0.001):
        super().__init__()
        self.convblock = VGGConvBlock(in_channels, batchnorm_momentum=batchnorm_momentum)
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features + self.convblock.CHANNEL_1 * 4, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
        )

    def forward(self, flat_obs, image):
        conv_values = self.convblock(image)
        features = torch.cat([flat_obs, conv_values.flatten(-3)], dim=-1)
        shape = features.shape
        if len(shape) == 3:
            features = features.flatten(0, 1)
            values = self.net(features)
            return values.unflatten(0, shape[0:2])
        else:
            return self.net(features)


class Policy(nn.Module):
    """
    Policy network for flat observation
    """

    def __init__(self, in_features, in_channels, hidden_dim, actions_n, batchnorm_momentum=0.001):
        super().__init__()
        self.convblock = VGGConvBlock(in_channels, batchnorm_momentum=batchnorm_momentum)
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features + self.convblock.CHANNEL_1 * 4, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=actions_n, bias=False)
        )

    def forward(self, flat_obs, image):
        conv_values = self.convblock(image)
        features = torch.cat([flat_obs, conv_values.flatten(-3)], dim=-1)
        shape = features.shape
        if len(shape) == 3:
            features = features.flatten(0, 1)
            # torch.cuda.memory._record_memory_history()
            logits = log_softmax(self.net(features), dim=-1)
            # torch.cuda.memory._dump_snapshot("policy_convnet_snapshot.pickle")
            return logits.unflatten(0, shape[0:2])
        else:
            return log_softmax(self.net(features), dim=-1)


IN_KEYS = ['flat_obs', 'ego_image']
EGO_PATCH_RADIUS = 4


def make_policy_module(policy_net, in_keys, device):
    in_keys = [in_keys] if isinstance(in_keys, str) else in_keys
    policy_module = TensorDictModule(
        policy_net,
        in_keys=in_keys,
        out_keys=['logits'],
    )

    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=['logits'],
        out_keys=['action'],
        distribution_class=Categorical,
        return_log_prob=True
    ).to(device)

    return policy_module



def train(args):
    """
    Optimize the agent using Proximal Policy Optimization (Actor - Critic)
    the Generalized Advantage Estimation module is used to compute Advantage
    """

    hparams = vars(args)
    try:
        hparams['version'] = version('superpacman')
    except PackageNotFoundError:
        pass

    exp_name = HRID().generate()
    if args.logger == 'csv':
        logger = CSVLogger(exp_name, args.logger, video_format='mp4', video_fps=3)
    else:
        logger = get_logger(args.logger, args.logger, experiment_name=exp_name)

    frames_per_batch = args.env_batch_size * args.steps_per_batch
    total_frames = args.env_batch_size * args.steps_per_batch * args.train_steps

    # environments
    env = make_env(args.env_batch_size, IN_KEYS,
                   device=args.device, ego_patch_radius=EGO_PATCH_RADIUS, seed=args.seed,
                   max_steps=args.max_steps_per_traj)
    check_env_specs(env)
    eval_env = make_env(32, IN_KEYS,
                        device=args.device, ego_patch_radius=EGO_PATCH_RADIUS, seed=args.seed,
                        max_steps=args.max_steps_per_traj)

    # networks
    in_features = env.observation_spec['flat_obs'].shape[-1]
    in_channels = env.observation_spec['ego_image'].shape[-3]
    actions_n = env.action_spec.n

    value_net = Value(in_features=in_features, in_channels=in_channels, hidden_dim=args.hidden_dim)
    policy_net = Policy(in_features=in_features, in_channels=in_channels, hidden_dim=args.hidden_dim, actions_n=actions_n)

    value_params = sum(p.numel() for p in value_net.parameters() if p.requires_grad)
    policy_params = sum(p.numel() for p in value_net.parameters() if p.requires_grad)
    hparams['value_params'] = value_params
    hparams['policy_params'] = policy_params
    print(f'value params: {value_params} policy_params {policy_params}')
    logger.log_hparams(hparams)

    value_module = ValueOperator(
        module=value_net,
        in_keys=IN_KEYS,
    ).to(args.device)

    policy_module = make_policy_module(policy_net, IN_KEYS, args.device)

    # no need to reuse data for PPO as it's an online algo
    # so we will go with datacollector only and collect fresh batches each time

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=args.device,
    )

    advantage_module = GAE(
        gamma=args.gamma, lmbda=args.lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=args.clip_epsilon,
        entropy_bonus=bool(args.entropy_eps),
        entropy_coef=args.entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1"
    )

    optim = Adam(loss_module.parameters(), lr=args.lr)

    def warmup(current_step: int):
        if current_step < args.warmup_steps:
            return 0.
        else:
            return 1.

    scheduler = LambdaLR(optim, lr_lambda=warmup)

    def save_checkpoint(filename):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "value_net_state_dict": value_net.state_dict(),
            "policy_net_state_dict": policy_net.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "power": args.power,
            "hidden_dim": args.hidden_dim
        }, filename)

    def load_checkpoint(filename):
        chkpt = torch.load(filename)
        power, hidden_dim = chkpt["power"], chkpt["hidden_dim"]
        assert chkpt["power"] == args.power, f"set --power flag to --power {power}"
        assert chkpt["hidden_dim"] == args.hidden_dim, f"set --hidden_dim flag to --hidden_dim {hidden_dim}"
        value_net.load_state_dict(chkpt["value_net_state_dict"])
        policy_net.load_state_dict(chkpt["policy_net_state_dict"])
        optim.load_state_dict(chkpt["optim_state_dict"])
        scheduler.load_state_dict(chkpt["scheduler_state_dict"])

    if args.load_checkpoint:
        load_checkpoint(args.load_checkpoint)

    # ready to start the training loop
    pbar = tqdm.tqdm(total=total_frames)
    train_reward_mean, train_reward_max, eval_reward_mean = 0., 0., 0.
    after_update = time()

    # torch.cuda.memory._record_memory_history()
    for i, tensordict_data in enumerate(collector):

        after_collect = time()
        env_time = after_collect - after_update

        # PPO update
        for _ in range(args.ppo_steps):
            advantage_module(tensordict_data)
            loss_vals = loss_module(tensordict_data)
            loss_value = (loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"])
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm)
            optim.step()
            optim.zero_grad()

        scheduler.step()
        after_update = time()
        update_time = after_update - after_collect

        # and now for the logging
        def retrieve_episode_stats(tensordict_data, loss_value, prefix=None):
            with torch.no_grad():
                prefix = '' if prefix is None else f"{prefix}_"
                episode_reward = tensordict_data["next", "episode_reward"]
                step_count = tensordict_data["step_count"].float()
                state_value = tensordict_data['state_value']
                entropy = loss_value['entropy']
                return {
                    f"{prefix}episode_reward_mean": episode_reward.mean().item(),
                    f"{prefix}episode_reward_max": episode_reward.max().item(),
                    f"{prefix}episode_reward_stdev": torch.std(episode_reward).item(),
                    f"{prefix}step_count_max": step_count.max().item(),
                    f"{prefix}step_count_mean": step_count.mean().item(),
                    f"{prefix}state_value_max": state_value.max().item(),
                    f"{prefix}state_value_mean": state_value.mean().item(),
                    f"{prefix}state_value_min": state_value.min().item(),
                    f"{prefix}policy_entropy_value_max": entropy.max().item(),
                    f"{prefix}policy_entropy_value_mean": entropy.mean().item(),
                    f"{prefix}policy_entropy_value_min": entropy.min().item(),
                }

        if i % 10 == 0:
            epi_stats = retrieve_episode_stats(tensordict_data, loss_vals, 'train')
            train_stats = {
                "train_learning rate": scheduler.get_last_lr()[0],
                "train_env_time": env_time,
                "train_update_time": update_time,
            }
            stats = {**epi_stats, **train_stats}
            for name, value in stats.items():
                logger.log_scalar(name, value, step=i)

            train_reward_mean = epi_stats['train_episode_reward_mean']
            train_reward_max = epi_stats['train_episode_reward_max']

        pbar.set_description(
            f'train reward mean/max {train_reward_mean:.2f}/{train_reward_max:.2f} eval reward mean: {eval_reward_mean:.2f}')
        pbar.update(tensordict_data.numel())

        # evaluation
        if i % args.eval_freq == 0:
            with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
                pbar.set_description('starting eval rollout')
                eval_rollout = eval_env.rollout(args.eval_len, policy_module, break_when_any_done=False)
                pbar.set_description('computing stats')
                advantage_module(eval_rollout)
                loss = loss_module(eval_rollout)
                epi_stats = retrieve_episode_stats(eval_rollout, loss, prefix='eval')
                for name, value in epi_stats.items():
                    logger.log_scalar(name, value, step=i)

                eval_reward_mean = epi_stats['eval_episode_reward_mean']
                pbar.set_description(f'saving checkpoint {eval_reward_mean:.2f}')
                save_checkpoint(f'checkpoints/{exp_name}/checkpoint_{i // args.eval_freq}_{eval_reward_mean:.2f}.pt')

    pbar.close()
    # torch.cuda.memory._dump_snapshot("full_snapshot.pickle")

    # once training is done, write a video of the best policy we found
    def best_checkpt(directory):
        best_rew = -inf
        best_chkpt = None
        for chkp in Path(directory).glob('*.pt'):
            reward_str = chkp.name.split('_')[-1][:-3]
            try:
                rew = float(reward_str)
                if rew > best_rew:
                    best_rew = rew
                    best_chkpt = str(chkp)
            except ValueError:
                warn(f'{reward_str} is not a valid float')

        return best_chkpt, best_rew

    best_chkpt, best_reward = best_checkpt(f'checkpoints/{exp_name}')
    if best_chkpt is not None:
        rollout_checkpoint(best_chkpt, suffix=f'eval_{best_reward:.2f}', logger=logger, device=args.device,
                           seed=args.seed, len=args.eval_len)


def rollout_checkpoint(checkpoint_filename, suffix, logger, device='cpu', seed=42, len=400, max_steps_per_trajectory=None):
    with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():

        eval_env = make_env(32, IN_KEYS, device=device, ego_patch_radius=EGO_PATCH_RADIUS,
                            seed=seed, log_video=True, logger=logger, max_steps=max_steps_per_trajectory)

        in_features = eval_env.observation_spec['flat_obs'].shape[-1]
        in_channels = eval_env.observation_spec['ego_image'].shape[-3]
        actions_n = eval_env.action_spec.n
        chkpt = torch.load(checkpoint_filename)
        hidden_dim = chkpt["hidden_dim"]
        policy_net = Policy(in_features=in_features, in_channels=in_channels, hidden_dim=hidden_dim, actions_n=actions_n)
        policy_net.load_state_dict(chkpt["policy_net_state_dict"])
        policy_module = make_policy_module(policy_net, IN_KEYS, device)

        print(f"rolling out policy {suffix}")
        eval_env.rollout(len, policy_module, break_when_any_done=False)

        print(f"logging video to {logger.log_dir}/{logger.exp_name}")
        eval_env.video_recorder.dump(suffix=suffix)


def enjoy_checkpoint(args):
    filename = Path(args.checkpoint).name
    logger = CSVLogger(filename, './enjoy', video_format='mp4', video_fps=3)
    rollout_checkpoint(args.checkpoint, suffix=filename, device=args.device, logger=logger, seed=args.seed,
                       len=args.length, max_steps_per_trajectory=args.max_steps_per_traj)
