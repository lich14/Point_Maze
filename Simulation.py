import datetime
import numpy as np
import torch
import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn as nn
from Basic import Plus_Net
from SAC import SAC_continuous, SAC_discrete
from Utils import soft_update, hard_update


class Doit():

    def __init__(
            self,
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
            start_train_loop=5000,
            test_step_oneloop=10000,
            start_usenet_step=5000,
            buffer_capacity=1000000,
            length=1000000,
            writer=None,
    ):

        super().__init__()

        self.skills = [SAC_continuous(obs_dim, action_dim, i, load_path, args) for i in range(option_dim)]
        self.option_framework = SAC_discrete(obs_dim, option_dim, load_path, args)
        self.forward_net = Plus_Net([256, 256], obs_dim, obs_dim, option_dim, layer_norm=False).to(device).double()
        self.forward_net_target = Plus_Net([256, 256], obs_dim, obs_dim, option_dim, layer_norm=False).to(device).double()
        hard_update(self.forward_net_target, self.forward_net)

        self.forward_net_optimizer = torch.optim.Adam(self.forward_net.parameters(), lr=args.lr)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.env = env
        self.logging = logging
        self.fixstart = args.fixstart

        self.device = device
        self.csv_path = csv_path
        self.fig_path = fig_path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.option_dim = option_dim

        self.batch_size = args.batch_size
        self.length = length
        self.writer = writer

        self.test_step_oneloop = test_step_oneloop
        self.start_train_loop = start_train_loop
        self.train_step = 0
        self.test_step = 0
        self.start_usenet_step = start_usenet_step

        self.step_num = 0
        self.episode_num = 0
        self.pi_z = [0 for i in range(option_dim)]
        self.pi_cur_z = [0 for i in range(option_dim)]

        self.cmap = sns.color_palette("Set1", option_dim, 0.9)
        self.dense_reward_skills = args.dense_reward_skills
        self.dense_reward_options = args.dense_reward_options

    def writereward(self, reward, step, path):
        if os.path.isfile(self.csv_path + path):
            with open(self.csv_path + path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, reward])
        else:
            with open(self.csv_path + path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'reward'])
                csv_write.writerow([step, reward])

    def to_np(self, a):
        return a.to('cpu').detach().squeeze().numpy().tolist()

    def test_oracle(self):
        state = self.to_np(self.env.reset(goal=torch.tensor([4, -0.5])))
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
        self.env.maze.plot(ax)
        pic_data = []
        for i in range(100):
            plot_state = self.env._state['state']
            pic_data.append([plot_state[0], plot_state[1]])
            if i < 10:
                action = [0, -1]
            elif i < 20:
                action = [1, 0]
            else:
                action = -(plot_state - self.env._state['goal']) / 10

            state, reward, terminal = self.env.step(action)
            if terminal:
                print(self.env.is_success, i)
                break

        pic_data = np.array(pic_data)
        ax.plot(pic_data[:, 0], pic_data[:, 1], color=self.cmap[1])
        fig.savefig(f'./oracle.pdf', dpi=300, bbox_inches='tight')

    def episode(self):
        start_state = None
        if self.fixstart:
            start_state = torch.tensor([0, -0.5])
        state = self.to_np(self.env.reset(state=start_state, goal=torch.tensor([4, -0.5])))
        self.env._done_on_success = False

        is_initial_states = 1
        steps = 0
        pre_option = np.random.randint(self.option_dim)

        while True:
            self.step_num += 1
            steps += 1

            if self.step_num > self.start_usenet_step:
                option = self.option_framework.select_option(state, pre_option, is_initial_states)
                option = self.to_np(option)
                action = self.skills[option].select_action(state)
            else:
                option = np.random.randint(self.option_dim)
                action = np.random.random(self.action_dim) * 2 - 1

            self.pi_z[option] += 1
            self.pi_cur_z[option] += 1

            next_state, reward, terminal = self.env.step(action)
            next_state = self.to_np(next_state)

            skills_reward = reward[self.dense_reward_skills]
            options_reward = reward[self.dense_reward_options]
            mask_terminal = False

            if terminal:
                mask_terminal = False if steps >= self.env.n else True

            self.skills[option].memory.push(state, action, skills_reward, next_state, int(mask_terminal))
            self.option_framework.memory.push(state, option, options_reward, next_state, int(mask_terminal))
            state, pre_option, is_initial_states = next_state, option, 0
            if terminal:
                self.episode_num += 1
                print(f'episode: {self.episode_num:<4}  success: {self.env.is_success}')
                break

    def compute_logprob_under_latent(self, s, s_, z, return_loss=False):
        """ Compute p(s'|z, s) for z """
        estimate_s_ = self.forward_net(s, z)
        estimate_target_s_ = self.forward_net_target(s, z)
        loss = self.mse_loss(estimate_s_, s_)
        loss_target = self.mse_loss(estimate_target_s_, s_)
        logprob = -1. * loss_target.sum(dim=1)
        loss_update = loss.sum(dim=1).mean() if return_loss else None
        return logprob, loss_update

    def compute_forward_MIreward(self, s, s_, z):
        z_extend = torch.tensor(z).type_as(s).long().expand(self.batch_size, 1)
        log_q_s_z, loss_update_forward_net = self.compute_logprob_under_latent(s, s_, z_extend, True)
        with torch.no_grad():
            sum_q_s_z_i = torch.zeros_like(log_q_s_z)
            for z_sample in range(self.option_dim):
                z_vector = torch.ones_like(z_extend) * z_sample
                sum_q_s_z_i += torch.exp(self.compute_logprob_under_latent(s, s_, z_vector)[0].detach())
        MI_forward = log_q_s_z.clamp(-1e10).detach() + torch.log(torch.tensor(self.option_dim).type_as(s).clamp(1e-10)) - torch.log(
            sum_q_s_z_i.clamp(1e-10))

        self.forward_net_optimizer.zero_grad()
        loss_update_forward_net.backward()
        nn.utils.clip_grad_norm_(self.forward_net.parameters(), 0.1)
        self.forward_net_optimizer.step()

        return MI_forward

    def compute_reverse_MIreward(self, s, z, a):
        fix_action_pi = torch.cat([self.skills[i].policy.get_logp(s, a).sum(dim=1, keepdim=True) for i in range(self.option_dim)], dim=1)
        option_selection_pi = self.option_framework.critic.sample_option(s)
        matrix_pi = (fix_action_pi * option_selection_pi).clamp(1e-10)
        sum_pi = matrix_pi.sum(dim=1, keepdim=True)
        cur_pi = (matrix_pi[:, z].view(-1, 1) / sum_pi).clamp(1e-10)

        return torch.log(cur_pi)

    def train(self):
        #if self.replay_buffer.num_transition >= self.start_train_loop and self.replay_buffer.num_transition % 1000 == 999:
        if self.step_num - self.train_step >= self.start_train_loop:
            for tini_train in range(50):
                self.option_framework.update_parameters(self.batch_size, self.writer, self.logging)
            for tini_train in range(200):
                for i in range(self.option_dim):
                    state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.skills[i].memory.sample(
                        batch_size=self.batch_size)
                    state_batch = torch.tensor(state_batch).double().to(self.device)
                    next_state_batch = torch.tensor(next_state_batch).double().to(self.device)
                    action_batch = torch.tensor(action_batch).double().to(self.device)
                    reward_batch = torch.tensor(reward_batch).double().to(self.device).unsqueeze(1)
                    mask_batch = torch.tensor(mask_batch).double().to(self.device).unsqueeze(1)

                    forward_MI_reward = self.compute_forward_MIreward(state_batch, next_state_batch, i).view(-1, 1)
                    reverse_MI_reward = self.compute_reverse_MIreward(state_batch, i, action_batch).view(-1, 1)

                    reward_batch = reward_batch + forward_MI_reward.detach() + reverse_MI_reward.detach()

                    qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs = self.skills[i].update_parameters(
                        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, self.train_step)
                    self.logging.info(
                        f'id: {i}, qf1_loss: {qf1_loss}, qf2_loss: {qf2_loss}, policy_loss: {policy_loss}, alpha: {alpha_tlogs}')

                    if self.writer and tini_train % 10 == 9:
                        self.writer.add_scalar(f'train_skills{i}/qf1_loss', qf1_loss, tini_train + self.train_step)
                        self.writer.add_scalar(f'train_skills{i}/qf2_loss', qf2_loss, tini_train + self.train_step)
                        self.writer.add_scalar(f'train_skills{i}/policy_loss', policy_loss, tini_train + self.train_step)
                        self.writer.add_scalar(f'train_skills{i}/alpha_loss', alpha_loss, tini_train + self.train_step)
                        self.writer.add_scalar(f'train_skills{i}/alpha', alpha_tlogs, tini_train + self.train_step)

            hard_update(self.forward_net_target, self.forward_net)
            self.train_step += 200
            if self.train_step % 50000 == 0:
                self.option_framework.save_model(self.train_step)
                for i in range(self.option_dim):
                    self.skills[i].save_model(self.train_step)
            print("----------------------------------------------------------------------")
            print(f"Skills Train step: {self.train_step}")
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("----------------------------------------------------------------------")

    def test_explore_option(self):
        episodes = 20
        self.env._done_on_success = True
        returns = np.zeros((episodes,), dtype=np.float32)
        pic_data = []
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
        self.env.maze.plot(ax)
        success_rate = 0

        for i in range(episodes):
            state = self.to_np(self.env.reset(state=torch.tensor([0, -0.5]), goal=torch.tensor([4, -0.5])))
            pre_option = np.random.randint(self.option_dim)
            plot_state = self.env._state['state']

            pic_data.append([plot_state[0], plot_state[1], pre_option])
            ax.plot(pic_data[0][0], pic_data[0][1], marker='o', markersize=8, color='black', zorder=11)

            episode_reward = 0.
            is_initial_states = 1
            while True:
                option = self.option_framework.select_option(state, pre_option, is_initial_states)
                option = self.to_np(option)
                action = self.skills[option].select_action(state, eval=True)
                if option != pre_option:
                    pic_data = np.array(pic_data)
                    ax.plot(pic_data[:, 0], pic_data[:, 1], color=self.cmap[pre_option])
                    pic_data = [[pic_data[-1, 0], pic_data[-1, 1], option]]
                state, reward, terminal = self.env.step(action)
                state = self.to_np(state)
                plot_state = self.env._state['state']
                episode_reward += reward[1]
                pic_data.append([plot_state[0], plot_state[1], option])
                if terminal:
                    success_rate += float(self.env.is_success) / episodes
                    pic_data = np.array(pic_data)
                    ax.plot(pic_data[:, 0], pic_data[:, 1], color=self.cmap[pre_option])
                    pic_data = []
                    break
                pre_option = option
                is_initial_states = 0
            returns[i] = episode_reward

        fig.savefig(f'./{self.fig_path}/explore_option{self.step_num}.pdf', dpi=300, bbox_inches='tight')
        mean_return = np.mean(returns)
        if self.writer:
            self.writer.add_scalar('explore/dense_reward', mean_return, self.step_num)
            self.writer.add_scalar('explore/success_rate', success_rate, self.step_num)

        print("----------------------------------------------------------------------")
        print("Test(explore option) Steps: {}, Avg. Reward: {}, Success rate: {}".format(self.step_num, mean_return, success_rate))
        print("----------------------------------------------------------------------")
        print(f"short option prob:{np.array(self.pi_cur_z) / sum(self.pi_cur_z)}")
        print(f"full option prob:{np.array(self.pi_z) / sum(self.pi_z)}")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("----------------------------------------------------------------------")
        self.pi_cur_z = [0 for i in range(self.option_dim)]

    def test_exploit_option(self):
        episodes = 20
        self.env._done_on_success = True
        returns = np.zeros((episodes,), dtype=np.float32)
        pic_data = []
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
        self.env.maze.plot(ax)
        success_rate = 0

        for i in range(episodes):
            state = self.to_np(self.env.reset(state=torch.tensor([0, -0.5]), goal=torch.tensor([4, -0.5])))
            pre_option = np.random.randint(self.option_dim)
            plot_state = self.env._state['state']

            pic_data.append([plot_state[0], plot_state[1], pre_option])
            ax.plot(pic_data[0][0], pic_data[0][1], marker='o', markersize=8, color='black', zorder=11)

            episode_reward = 0.
            is_initial_states = 1
            while True:
                option = self.option_framework.exploit_option(state, pre_option, is_initial_states)
                option = self.to_np(option)
                action = self.skills[option].select_action(state, eval=True)
                if option != pre_option:
                    pic_data = np.array(pic_data)
                    ax.plot(pic_data[:, 0], pic_data[:, 1], color=self.cmap[pre_option])
                    pic_data = [[pic_data[-1, 0], pic_data[-1, 1], option]]
                state, reward, terminal = self.env.step(action)
                state = self.to_np(state)
                plot_state = self.env._state['state']
                episode_reward += reward[1]
                pic_data.append([plot_state[0], plot_state[1], option])
                if terminal:
                    success_rate += float(self.env.is_success) / episodes
                    pic_data = np.array(pic_data)
                    ax.plot(pic_data[:, 0], pic_data[:, 1], color=self.cmap[pre_option])
                    pic_data = []
                    break
                pre_option = option
                is_initial_states = 0
            returns[i] = episode_reward

        fig.savefig(f'./{self.fig_path}/exploit_option{self.step_num}.pdf', dpi=300, bbox_inches='tight')
        mean_return = np.mean(returns)
        if self.writer:
            self.writer.add_scalar('exploit/success_rate', success_rate, self.step_num)

        print("----------------------------------------------------------------------")
        print("Test(exploit option) Steps: {}, Avg. Reward: {}, Success rate: {}".format(self.step_num, mean_return, success_rate))
        print("----------------------------------------------------------------------")

    def test_different_options(self):
        point = [[0, -0.5], [0, -3.5], [4, -3.5]]
        for kk in range(3):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
            self.env.maze.plot(ax)

            for option_choose in range(0, self.option_dim):
                episodes = 2
                pic_data = []

                for _ in range(episodes):
                    state = self.to_np(self.env.reset(state=torch.tensor(point[kk]), goal=torch.tensor([4, -0.5])))
                    pre_option = np.random.randint(self.option_dim)
                    plot_state = self.env._state['state']

                    pic_data.append([plot_state[0], plot_state[1], pre_option])
                    ax.plot(pic_data[0][0], pic_data[0][1], marker='o', markersize=8, color='black', zorder=11)

                    while True:
                        option = option_choose
                        action = self.skills[option].select_action(state, eval=True)
                        if option != pre_option:
                            pic_data = np.array(pic_data)
                            ax.plot(pic_data[:, 0], pic_data[:, 1], color=self.cmap[pre_option])
                            pic_data = [[pic_data[-1, 0], pic_data[-1, 1], option]]
                        state, _, terminal = self.env.step(action)
                        state = self.to_np(state)
                        plot_state = self.env._state['state']
                        pic_data.append([plot_state[0], plot_state[1], option])
                        if terminal:
                            pic_data = np.array(pic_data)
                            ax.plot(pic_data[:, 0], pic_data[:, 1], color=self.cmap[pre_option])
                            pic_data = []
                            break
                        pre_option = option

            fig.savefig(f'./{self.fig_path}/diff_option{self.step_num}_{kk}.pdf', dpi=300, bbox_inches='tight')

    def run(self):
        while True:
            self.episode()
            self.train()
            if self.step_num - self.test_step >= self.test_step_oneloop:
                self.test_exploit_option()
                self.test_explore_option()
                self.test_different_options()
                self.test_step += self.test_step_oneloop

            if self.step_num >= self.length:
                break
