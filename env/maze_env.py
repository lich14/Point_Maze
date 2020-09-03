# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import numpy as np
from env.mazes import mazes_dict, make_crazy_maze, make_experiment_maze, make_hallway_maze, make_u_maze


class Env:

    def __init__(
            self,
            n=None,
            maze_type=None,
            use_antigoal=False,
            ddiff=False,
            ignore_reset_start=False,
            done_on_success=True,
            inverse_reward=True,
    ):
        self.n = n
        self.inverse_reward = inverse_reward

        self._mazes = mazes_dict
        self.maze_type = maze_type.lower()

        self._ignore_reset_start = bool(ignore_reset_start)
        self._done_on_success = bool(done_on_success)

        # Generate a crazy maze specified by its size and generation seed
        if self.maze_type.startswith('crazy'):
            _, size, seed = self.maze_type.split('_')
            size = int(size)
            seed = int(seed)
            self._mazes[self.maze_type] = {'maze': make_crazy_maze(size, seed), 'action_range': 0.95}

        # Generate an "experiment" maze specified by its height, half-width, and size of starting section
        if self.maze_type.startswith('experiment'):
            _, h, half_w, sz0 = self.maze_type.split('_')
            h = int(h)
            half_w = int(half_w)
            sz0 = int(sz0)
            self._mazes[self.maze_type] = {'maze': make_experiment_maze(h, half_w, sz0), 'action_range': 0.25}

        if self.maze_type.startswith('corridor'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_hallway_maze(corridor_length), 'action_range': 0.95}

        if self.maze_type.startswith('umaze'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_u_maze(corridor_length), 'action_range': 0.95}

        assert self.maze_type in self._mazes

        self.use_antigoal = bool(use_antigoal)
        self.ddiff = bool(ddiff)

        self._state = dict(s0=None, prev_state=None, state=None, goal=None, n=None, done=None, d_goal_0=None, d_antigoal_0=None)

        self.dist_threshold = 0.15

        self.reset()

    @property
    def state_size(self):
        return 2

    @property
    def goal_size(self):
        return 2

    @property
    def action_size(self):
        return 2

    @staticmethod
    def to_tensor(x):
        return torch.FloatTensor(x)

    @staticmethod
    def to_coords(x):
        if isinstance(x, (tuple, list)):
            return x[0], x[1]
        if isinstance(x, torch.Tensor):
            x = x.data.numpy()
        return float(x[0]), float(x[1])

    @staticmethod
    def dist(goal, outcome):
        # return torch.sum(torch.abs(goal - outcome))
        return torch.sqrt(torch.sum(torch.pow(goal - outcome, 2)))

    @property
    def maze(self):
        return self._mazes[self.maze_type]['maze']

    @property
    def action_range(self):
        return self._mazes[self.maze_type]['action_range']

    @property
    def state(self):
        return self._state['state'].view(-1).detach()

    @property
    def goal(self):
        return self._state['goal'].view(-1).detach()

    @property
    def antigoal(self):
        return self._state['antigoal'].view(-1).detach()

    def get_obs(self, point):
        radar = 10 * np.ones((8,), dtype=np.float32)
        # here 10 is assume to be maximum maze size from corner to corner
        # right left, up down, up right and down left, down right and up left
        for x, y in self.maze._walls:
            if x[0] == x[1]:
                for i in range(3):
                    # i:0 right and left
                    # i:1 up right and down left
                    # i:2 down right and up left
                    if i == 0 and point[1] <= max(y) and point[1] >= min(y):
                        distance = abs(x[0] - point[0])
                        if x[0] > point[0] and distance < radar[0]:
                            radar[0] = distance
                            # the wall is on the right of the point, calculate right distance
                        if x[0] <= point[0] and distance < radar[1]:
                            radar[1] = distance
                            # the wall is on the left of the point, calculate left distance

                    if i == 1:
                        deta = point[1] + (x[0] - point[0])
                        if deta <= max(y) and deta >= min(y):
                            distance = pow(2, 0.5) * abs(x[0] - point[0])
                            if x[0] > point[0] and distance < radar[4]:
                                # the wall is on the up right of the point
                                radar[4] = distance
                            if x[0] <= point[0] and distance < radar[5]:
                                # the wall is on the down left of the point
                                radar[5] = distance
                    if i == 2:
                        deta = point[1] - (x[0] - point[0])
                        if deta <= max(y) and deta >= min(y):
                            distance = pow(2, 0.5) * abs(x[0] - point[0])
                            if x[0] > point[0] and distance < radar[6]:
                                # the wall is on the down right of the point
                                radar[6] = distance
                            if x[0] <= point[0] and distance < radar[7]:
                                # the wall is on the up left of the point
                                radar[7] = distance

            else:
                for i in range(3):
                    # i:0 up down
                    # i:1 up right and down left
                    # i:2 down right and up left
                    if i == 0 and point[0] <= max(x) and point[0] >= min(x):
                        distance = abs(y[0] - point[1])
                        if y[0] > point[1] and distance < radar[2]:
                            radar[2] = distance
                            # the wall is on the up of the point
                        if y[0] <= point[1] and distance < radar[3]:
                            radar[3] = distance
                            # the wall is on the down of the point

                    if i == 1:
                        deta = point[0] + (y[0] - point[1])
                        distance = pow(2, 0.5) * abs(y[0] - point[1])
                        if deta <= max(x) and deta >= min(x):
                            if y[0] > point[1] and distance < radar[4]:
                                # the wall is on the up right of the point
                                radar[4] = distance
                            if y[0] <= point[1] and distance < radar[5]:
                                # the wall is on the down left of the point
                                radar[5] = distance
                    if i == 2:
                        deta = point[0] - (y[0] - point[1])
                        distance = pow(2, 0.5) * abs(y[0] - point[1])
                        if deta <= max(x) and deta >= min(x):
                            if y[0] < point[1] and distance < radar[6]:
                                # the wall is on the down right of the point
                                radar[6] = distance
                            if y[0] >= point[1] and distance < radar[7]:
                                # the wall is on the up left of the point
                                radar[7] = distance

        radar = (radar - 2) / 2
        return radar

    def calculate_oracle_reward(self, state):
        if (self.goal - torch.tensor([4.0000, -0.5000])).mean() != 0:
            raise (f'error goal: {self.goal}')

        if state[0] <= 2.5 and state[0] >= 1.5 and state[1] >= -2.5 and state[1] <= 0.5:
            reward = -10.5 - self.dist(state, torch.tensor([1.5, -2.5])).to('cpu').numpy()
        elif state[0] <= 1.5 and state[0] >= 0.5 and state[1] >= -2.5 and state[1] <= -1.5:
            reward = -9.5 - self.dist(state, torch.tensor([0.5, -2.5])).to('cpu').numpy()
        elif state[0] <= 0.5 and state[0] >= -0.5 and state[1] >= -4.5 and state[1] <= 0.5:
            reward = -7 - self.dist(state, torch.tensor([0.5, -4])).to('cpu').numpy()
        elif state[0] <= 4.5 and state[0] >= 0.5 and state[1] >= -4.5 and state[1] <= -3.5:
            reward = -3 - self.dist(state, torch.tensor([4, -3.5])).to('cpu').numpy()
        elif state[0] <= 4.5 and state[0] >= 3.5 and state[1] >= -3.5 and state[1] <= 0.5:
            reward = -self.dist(state, torch.tensor([4, -0.5])).to('cpu').numpy()
        else:
            reward = -100

        return reward

    @property
    def reward(self):
        r_sparse = 2 * float(self.is_success)
        r_dense = -self.dist(self.goal, self.state).to('cpu').squeeze().numpy()

        if self.inverse_reward:
            return r_sparse + r_dense / 7, r_sparse + r_dense / 7, self.calculate_oracle_reward(self.state) / 12 + r_sparse
        else:
            return r_sparse + r_dense / 7, r_sparse - 0.1, self.calculate_oracle_reward(self.state) / 12 + r_sparse

    @property
    def achieved(self):
        return self.goal if self.is_success else self.state

    @property
    def is_done(self):
        return bool(self._state['done'])

    @property
    def is_success(self):
        d = self.dist(self.goal, self.state)
        return d <= self.dist_threshold

    @property
    def d_goal_0(self):
        return self._state['d_goal_0'].item()

    @property
    def d_antigoal_0(self):
        return self._state['d_antigoal_0'].item()

    @property
    def next_phase_reset(self):
        return {'state': self._state['s0'].detach(), 'goal': self.goal, 'antigoal': self.achieved}

    @property
    def sibling_reset(self):
        return {'state': self._state['s0'].detach(), 'goal': self.goal}

    def reset(self, state=None, goal=None, antigoal=None):
        if state is None or self._ignore_reset_start:
            s_xy = self.to_tensor(self.maze.sample_start())
        else:
            s_xy = self.to_tensor(state)
        if goal is None:
            if 'square' in self.maze_type:
                g_xy = self.to_tensor(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold))
            else:
                g_xy = self.to_tensor(self.maze.sample_goal())
        else:
            g_xy = self.to_tensor(goal)

        if antigoal is None:
            ag_xy = self.to_tensor(g_xy)
        else:
            ag_xy = self.to_tensor(antigoal)

        self._state = {
            's0': s_xy,
            'prev_state': s_xy * torch.ones_like(s_xy),
            'state': s_xy * torch.ones_like(s_xy),
            'goal': g_xy,
            'antigoal': ag_xy,
            'n': 0,
            'done': False,
            'd_goal_0': self.dist(g_xy, s_xy),
            'd_antigoal_0': self.dist(g_xy, ag_xy),
        }

        return torch.cat([self._state['state'], torch.tensor(self.get_obs(s_xy)), g_xy], dim=-1)

    def step(self, action):
        try:
            next_state = self.maze.move(self.to_coords(self._state['state']), self.to_coords(action))
        except:
            print('state', self.to_coords(self._state['state']))
            print('action', self.to_coords(action))
            raise
        self._state['prev_state'] = self.to_tensor(self._state['state'])
        self._state['state'] = self.to_tensor(next_state)
        self._state['n'] += 1
        done = self._state['n'] >= self.n
        if self._done_on_success:
            done = done or self.is_success
        self._state['done'] = done

        return torch.cat([self._state['state'], torch.tensor(self.get_obs(self._state['state'])), self._state['goal']],
                         dim=-1), self.reward, done

    def sample(self):
        return self.maze.sample()
