from gridworld import Gridworld, RGBFullObsTransform
from time import time
from torchrl.envs import TransformedEnv, FlattenObservation, CatTensors, StepCounter, RewardSum, check_env_specs, ExplorationType, set_exploration_type
from torchrl.record import VideoRecorder, CSVLogger
import torch


def make_env(log_video, env_batch_size):
    env = Gridworld(batch_size=torch.Size([env_batch_size]), device='cuda')
    env = TransformedEnv(
        env
    )
    env.append_transform(
        FlattenObservation(-2, -1,
                           in_keys=["player_tiles", "wall_tiles", "reward_tiles", 'terminal_tiles'],
                           out_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles", 'flat_terminal_tiles']
                           )
    )
    env.append_transform(
        CatTensors(
            in_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles", 'flat_terminal_tiles'],
            out_key='flat_obs'
        )
    )
    env.append_transform(StepCounter())
    env.append_transform(RewardSum(reset_keys=['_reset']))
    recorder = None
    if log_video:
        env.append_transform(RGBFullObsTransform())
        logger = CSVLogger(exp_name='performance', log_dir="logs", video_fps=3, video_format='mp4')
        recorder = VideoRecorder(logger=logger, tag='pacman', fps=3, skip=1)
        env.append_transform(recorder)
    check_env_specs(env)
    env.set_seed(42)
    return env, recorder


def test_performance():

    with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():

        eval_env, eval_recorder = make_env(True, 32)
        start_time = time()
        eval_rollout = eval_env.rollout(1000, break_when_any_done=False)
        rollout_done = time()
        eval_recorder.dump(suffix=f'performance')
        video_dump_done = time()

        print(f'rollout_time {rollout_done - start_time} dump_time {video_dump_done - rollout_done}')

    with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
        eval_env, eval_recorder = make_env(False, 32)
        start_time = time()
        eval_rollout = eval_env.rollout(1000, break_when_any_done=False)
        rollout_done = time()

        print(f'rollout_time {rollout_done - start_time}')

