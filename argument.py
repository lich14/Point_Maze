import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="Ant-v2", help='Mujoco Gym environment (default: Ant-v2)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.16, metavar='G')
    parser.add_argument('--automatic_entropy_tuning_high', action='store_true', default=False)
    parser.add_argument('--automatic_entropy_tuning_low', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N', help='Steps sampling random actions (default: 10000)')
    parser.add_argument(
        '--target_update_interval', type=int, default=1, metavar='N', help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N', help='size of replay buffer (default: 10000000)')

    parser.add_argument('--GPU', type=str, default="cuda:1", help='bool')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--option-dim', type=int, default=4, help='random seed')
    parser.add_argument('--Beta-add', type=float, default=0.01)
    parser.add_argument('--POPArt', action='store_false', default=True)
    parser.add_argument('--fixstart', action='store_true', default=False)
    parser.add_argument('--inverse-reward', action='store_false', default=True)
    parser.add_argument('--dense-reward-skills', type=int, default=0)
    parser.add_argument('--dense-reward-options', type=int, default=1)
    parser.add_argument('--MI-forward-weight', type=float, default=0., help='change the weight of mutual information')
    parser.add_argument('--MI-reverse-weight', type=float, default=0., help='change the weight of mutual information')
    parser.add_argument('--beta-weight', type=float, default=0.1)
    parser.add_argument('--sr-bound', type=float, default=5.0)
    parser.add_argument('--JS-lamda', type=float, default=0, help='change the weight of regular term while updating option policy')
    parser.add_argument('--length', type=int, default=1000000, help='running length')

    args = parser.parse_args()

    return args
