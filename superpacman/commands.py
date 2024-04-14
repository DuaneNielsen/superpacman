from . import play
from . import train_ppo
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    play_parser = subparsers.add_parser('play', help='Run the game')
    play_parser.add_argument('--partial_radius', type=int, default=0, help='distance the agent can see')
    play_parser.set_defaults(func=play.play)

    train_ppo_parser = subparsers.add_parser('train', help='Train the agent')
    train_ppo_parser.add_argument('--exp_name', default='superpacman', help="experiment name")
    train_ppo_parser.add_argument('--device', default='cpu', help="cuda or cpu")
    train_ppo_parser.add_argument('--seed', type=int, default=42, help="seed (defaults 42)")
    train_ppo_parser.add_argument('--env_batch_size', type=int, default=2048, help="number of environments")
    train_ppo_parser.add_argument('--steps_per_batch', type=int, default=8, help="number of steps to take in env per batch")
    train_ppo_parser.add_argument('--train_steps', type=int, default=1000, help="number of PPO updates to run")
    train_ppo_parser.add_argument('--clip_epsilon', type=float, default=0.1, help="PPO clipping parameter")
    train_ppo_parser.add_argument('--gamma', type=float, default=0.99, help="GAE gamma parameter")
    train_ppo_parser.add_argument('--lmbda', type=float, default=0.99, help="GAE lambda parameter")
    train_ppo_parser.add_argument('--entropy_eps', type=float, default=0.08, help="policy entropy bonus weight")
    train_ppo_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="gradient clipping")
    train_ppo_parser.add_argument('--power', type=int, default=5, help="power of squeezenet")
    train_ppo_parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dim size of MLP")
    train_ppo_parser.add_argument('--lr', type=float, default=1e-3, help="Adam learning rate")
    train_ppo_parser.add_argument('--lr_sched_step_size', type=int, default=1e5, help="decay lr after this many steps")
    train_ppo_parser.add_argument('--lr_sched_gamma', type=int, default=0.7, help="decay lr after this many steps")
    train_ppo_parser.add_argument('--ppo_steps', type=int, default=6, help="number of ppo updates per batch")
    train_ppo_parser.add_argument('--eval_freq', type=int, default=100, help="run eval after this many training steps")
    train_ppo_parser.add_argument('--eval_len', type=int, default=400, help="run eval after this many training steps")
    train_ppo_parser.add_argument('--logger', choices=['csv', 'wandb', 'mlflow', 'tensorboard'], default='csv',
                                  help='supported loggers')
    train_ppo_parser.add_argument('--warmup_steps', type=int, default=16, help='delay before starting to learn')
    train_ppo_parser.add_argument('--load_checkpoint', help='load the checkpoint')
    train_ppo_parser.set_defaults(func=train_ppo.train)

    enjoy_parser = subparsers.add_parser('enjoy', help='Write a video of the policy in action')
    enjoy_parser.add_argument('checkpoint', type=str, help='checkpoint')
    enjoy_parser.add_argument('--device', type=str, default='cpu')
    enjoy_parser.add_argument('--seed', type=int, default=42)
    enjoy_parser.add_argument('--length', type=int, default=400, help='rollout length')
    enjoy_parser.set_defaults(func=train_ppo.enjoy_checkpoint)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        # If no subcommand is provided, print the help message
        parser.print_help()


if __name__ == '__main__':
    main()