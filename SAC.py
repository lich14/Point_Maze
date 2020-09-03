import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn as nn
from Utils import soft_update, hard_update, ReplayMemory
from Network import GaussianPolicy, QNetwork, TwinnedQNetwork, PopArt, Q_discrete_Network, Beta_network


class SAC_continuous(object):

    def __init__(self, obs_dim, action_dim, id, load_path, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.id = id
        self.memory = ReplayMemory(args.replay_size)
        self.load_path = load_path

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning_low

        use_cuda = torch.cuda.is_available()
        self.device = torch.device(args.GPU if use_cuda else "cpu")
        self.alpha = torch.tensor(args.alpha).to(self.device)

        self.critic = TwinnedQNetwork(obs_dim, action_dim, args.hidden_size).to(device=self.device).double()
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = TwinnedQNetwork(obs_dim, action_dim, args.hidden_size).to(device=self.device).double()

        hard_update(self.critic_target, self.critic)
        self.Q1_normer = PopArt(self.critic.Q1.last_fc)
        self.Q2_normer = PopArt(self.critic.Q2.last_fc)

        self.Q1_target_normer = PopArt(self.critic_target.Q1.last_fc)
        self.Q2_target_normer = PopArt(self.critic_target.Q2.last_fc)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.alpha = torch.tensor(1.).to(self.device)
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device, dtype=torch.double)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.policy = GaussianPolicy(obs_dim, action_dim, args.hidden_size).to(self.device).double()
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.tensor(state).double().to(self.device).unsqueeze(0)
        action, _ = self.policy.sample(state, deterministic=eval)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, updates):

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample(next_state_batch, return_log_prob=True)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            qf1_next_target_unnorm, qf2_next_target_unnorm = self.Q1_target_normer.unnorm(qf1_next_target), self.Q2_target_normer.unnorm(
                qf2_next_target)

            min_qf_next_target = torch.min(qf1_next_target_unnorm, qf2_next_target_unnorm) - self.alpha * next_state_log_pi
            next_q1_value = self.Q1_normer.update(reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target))
            next_q2_value = self.Q2_normer.update(reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target))

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q1_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q2_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi = self.policy.sample(state_batch, return_log_prob=True)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        qf1_pi_unnorm, qf2_pi_unnorm = self.Q1_normer.unnorm(qf1_pi), self.Q2_normer.unnorm(qf2_pi)
        min_qf_pi_unnorm = torch.min(qf1_pi_unnorm, qf2_pi_unnorm)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi_unnorm).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = 0.5 * (self.Q1_normer.norm(policy_loss) + self.Q2_normer.norm(policy_loss))

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = self.alpha  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.Q1_target_normer, self.Q1_normer, self.tau)
            soft_update(self.Q2_target_normer, self.Q2_normer, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, step):
        actor_path = self.load_path + "skills_actor_{}_{}.lch".format(self.id, step)
        critic_path = self.load_path + "skills_critic_{}_{}.lch".format(self.id, step)
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, step):
        print('Loading models')
        actor_path = self.load_path + "skills_actor_{}_{}.lch".format(self.id, step)
        critic_path = self.load_path + "skills_critic_{}_{}.lch".format(self.id, step)
        self.policy.load_state_dict(torch.load(actor_path, map_location='cpu'))
        self.critic.load_state_dict(torch.load(critic_path, map_location='cpu'))


class SAC_discrete(object):

    def __init__(self, obs_dim, option_dim, load_path, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.memory = ReplayMemory(args.replay_size)
        self.load_path = load_path
        self.obs_dim = obs_dim
        self.Beta_add = args.Beta_add
        self.beta_weight = args.beta_weight
        self.update_num = 0

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning_high

        use_cuda = torch.cuda.is_available()
        self.device = torch.device(args.GPU if use_cuda else "cpu")
        self.alpha = torch.tensor(args.alpha).to(self.device)

        self.critic = Q_discrete_Network(obs_dim, option_dim, args.hidden_size).to(device=self.device).double()
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = Q_discrete_Network(obs_dim, option_dim, args.hidden_size).to(device=self.device).double()

        hard_update(self.critic_target, self.critic)
        self.Q_normer = PopArt(self.critic.last_fc)
        self.Q_target_normer = PopArt(self.critic_target.last_fc)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(option_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device, dtype=torch.double)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.Beta = Beta_network(obs_dim, option_dim, args.hidden_size).to(self.device).double()
        self.Beta_optim = Adam(self.Beta.parameters(), lr=args.lr)

    def select_option(self, state, pre_option, ifinitial):
        state = torch.tensor(state).double().to(self.device)
        pre_option = torch.tensor(pre_option).long().to(self.device)
        ifinitial = torch.tensor(ifinitial).long().to(self.device)

        beta_withpreoption = self.Beta(state)[pre_option]
        q = self.critic.sample_option(state)
        q_rechoose = q.clone()
        q_rechoose[pre_option] = -1e20
        mask = torch.zeros_like(q_rechoose)
        mask[pre_option] = 1
        q_rechoose_softmax = torch.softmax(q_rechoose, dim=-1)

        pi = ifinitial * q + (1 - ifinitial) * ((1 - beta_withpreoption) * mask + beta_withpreoption * q_rechoose_softmax)
        dist = torch.distributions.Categorical(probs=pi)
        option = dist.sample()
        return option

    def exploit_option(self, state, pre_option, ifinitial):
        state = torch.tensor(state).double().to(self.device)
        pre_option = torch.tensor(pre_option).long().to(self.device)
        ifinitial = torch.tensor(ifinitial).long().to(self.device)

        beta_withpreoption = self.Beta(state)[pre_option]
        q = self.critic.sample_option(state)
        q_rechoose = q.clone()
        q_rechoose[pre_option] = -1e20
        mask = torch.zeros_like(q_rechoose)
        mask[pre_option] = 1
        q_rechoose_softmax = torch.softmax(q_rechoose, dim=-1)

        pi = ifinitial * q + (1 - ifinitial) * ((1 - beta_withpreoption) * mask + beta_withpreoption * q_rechoose_softmax)
        option = torch.argmax(pi)
        return option

    def update_parameters(self, batch_size, writer, logging):
        # Sample a batch from memory
        state_batch, option_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size=batch_size)

        state_batch = torch.tensor(state_batch).double().to(self.device)
        next_state_batch = torch.tensor(next_state_batch).double().to(self.device)
        option_batch = torch.tensor(option_batch).double().to(self.device).view(-1, 1).long()
        reward_batch = torch.tensor(reward_batch).double().to(self.device).view(-1, 1)
        done_batch = torch.tensor(done_batch).double().to(self.device).view(-1, 1)

        with torch.no_grad():
            qf_next = self.critic_target(next_state_batch)
            qf_next_unnorm = self.Q_target_normer.unnorm(qf_next)
            qf_next_update = self.alpha * torch.log(((qf_next_unnorm / self.alpha).exp()).sum(dim=1, keepdim=True).clamp(1e-10))
            next_q_value = self.Q_normer.update(reward_batch + (1 - done_batch) * self.gamma * qf_next_update)

        qf = self.critic(state_batch)
        qf_update = torch.gather(qf, 1, option_batch)
        qf_loss = F.mse_loss(qf_update, next_q_value)

        if self.update_num % 5 == 0:
            new_option_beta = self.Beta(state_batch)
            with torch.no_grad():
                minus_u_v = qf - (torch.softmax(qf, dim=-1) * qf).sum(dim=1, keepdim=True)
                minus_u_v = minus_u_v / torch.max(torch.abs(minus_u_v), dim=1, keepdim=True)[0]
                qf_next_update_beta = self.critic(next_state_batch)
                minus_s_s_ = torch.abs(qf_next_update_beta - qf) / torch.max(torch.abs(qf), torch.tensor(1).type_as(qf))
            beta_loss = (new_option_beta * (self.beta_weight * minus_u_v + (1 - self.beta_weight) * minus_s_s_ + self.Beta_add)).mean()
            logging.info(f"beta_loss_option: {beta_loss}, minus_u_v: {minus_u_v.mean()}, minus_s_s_: {minus_s_s_.mean()}")

            self.Beta_optim.zero_grad()
            beta_loss.backward()
            self.Beta_optim.step()

            writer.add_scalar('train_option/beta_loss', beta_loss.item(), self.update_num)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        logging.info(f"qf_loss_option: {qf_loss}")

        if self.automatic_entropy_tuning:
            pi_update_alpha = torch.softmax(qf, dim=1).detach()
            alpha_loss = -(pi_update_alpha * self.log_alpha *
                           (torch.log(pi_update_alpha.clamp(1e-10)) + self.target_entropy)).sum(dim=1).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = self.alpha  # For TensorboardX logs

        logging.info(f"alpha: {self.alpha}, loss: {alpha_loss}")
        writer.add_scalar('train_option/alpha_loss', alpha_loss.item(), self.update_num)
        writer.add_scalar('train_option/alpha', self.alpha.item(), self.update_num)

        if self.update_num % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.Q_target_normer, self.Q_normer, self.tau)

        writer.add_scalar('train_option/qf_loss', qf_loss.item(), self.update_num)
        self.update_num += 1

    # Save model parameters
    def save_model(self, step):
        beta_path = self.load_path + "beta_{}.lch".format(step)
        critic_path = self.load_path + "options_critic_{}.lch".format(step)
        torch.save(self.Beta.state_dict(), beta_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, step):
        print('Loading models')
        beta_path = self.load_path + "beta_{}.lch".format(step)
        critic_path = self.load_path + "options_critic_{}.lch".format(step)

        self.Beta.load_state_dict(torch.load(beta_path, map_location='cpu'))
        self.critic.load_state_dict(torch.load(critic_path, map_location='cpu'))
