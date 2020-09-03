import torch
import logging
import os
import gym
import numpy as np
import random
import time
import argparse
import gym
import mujoco_py
import mujoco
import matplotlib.pyplot as plt

from mujoco.create_maze_env import create_maze_env
from argument import get_args
from Simulation import Doit
from torch.utils.tensorboard import SummaryWriter


def logger_config(log_path, logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def experiment(args, device):
    '''env = create_maze_env(
        'PointMaze',
        maze_size_scaling=4,
        goal=np.array([0, 0]),
        dense_reward_type=2,
        goal_args=[[-4, -4], [20, 20]],
    )'''
    '''obs_dim = env.observation_space['observation'].shape[0] + 2
    action_dim = env.action_space.shape[0]
    action_multi = env.action_space.high
    option_dim = args.option_dim

    main_name = f"PointMaze2_MI_beta_weight_{args.beta_weight}_{args.MI_forward_weight}_{args.MI_reverse_weight}"
    exp_name = f"{main_name}/seed_{args.seed}"
    csv_path = f"{exp_name}/"
    load_path = f"{exp_name}/para/"
    log_path = f"{exp_name}/record.log"
    board_path = f"runs/{main_name}"
    fig_path = f"fig/{main_name}"
    writer = SummaryWriter(log_dir=board_path)

    for dir in [main_name, exp_name, load_path, 'fig', fig_path]:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    logging = logger_config(log_path, 'fly')
    task = Doit(
        args,
        env,
        obs_dim,
        action_dim,
        option_dim,
        device,
        csv_path,
        fig_path,
        load_path,
        logging,
        writer=writer,
        length=args.length,
    )'''
    env = gym.make('Hopper-v2')
    out = env.reset()
    env.render()

    terminal = False
    while (True):
        action = np.random.uniform(-1, 1, size=env.action_space.shape)
        print(action)
        time.sleep(0.2)
        env.step(action)
    #task.run()
    #task.test_explore_option()
    #task.test_oracle()


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    args = get_args()
    setup_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.GPU if use_cuda else "cpu")
    if use_cuda:
        if args.GPU == "cuda:0":
            torch.cuda.set_device(0)

        if args.GPU == "cuda:1":
            torch.cuda.set_device(1)

        if args.GPU == "cuda:2":
            torch.cuda.set_device(2)

        if args.GPU == "cuda:3":
            torch.cuda.set_device(3)

    experiment(args, device)
