import itertools
import time
import numpy as np
import os
import pandas as pd
import random
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
# from torch_geometric.data import Batch

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import gym
from gym.wrappers import Monitor

from collections import deque


def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)


use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda')
    print("using GPU")
else:
    device = torch.device('cpu')
    print("using CPU")


class A2C:
    def __init__(self, config, env, model, writer=None):
        """
        初始化强化学习代理。
        :param config: 包含配置参数的字典
        :param env: 环境对象，代理将在其中执行任务调度
        :param model: 强化学习模型对象，用于学习任务调度策略
        :param writer: 用于记录训练过程的写入器
        """

        self.config = config
        self.env = env
        make_seed(config['seed'])
        # self.env.seed(config['seed'])
        self.gamma = config['gamma']
        self.entropy_cost = config["entropy_coef"]
        self.noise = config['noise'] if 'noise' in config.keys() else config['env_settings']['noise']
        self.random_id = str(np.random.randint(0, 9, 10)).replace(' ', '_')

        self.network = model.to(device)
        if config["model_path"] is not None and config["model_path"] != 'none':
            # self.network.load_state_dict(torch.load(config['model_path']))
            self.network = torch.load(config['model_path'])

        if config['optimizer'] == "sgd":
            self.optimizer = optim.SGD(self.network.parameters(), config['lr'])
        elif config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=config['lr'])
        else:
            self.optimizer = optim.RMSprop(self.network.parameters(), config['lr'], eps=config['eps'])
        self.writer = writer

        if config['scheduler'] == 'cyclic':
            ratio = config['sched_ratio']
            self.scheduler = CyclicLR(self.optimizer, base_lr=config['lr']/ratio, max_lr=config['lr']*ratio,
                                      step_size_up=config['step_up'])
        elif config['scheduler'] == 'lambda':
            lambda2 = lambda epoch: 0.99 ** epoch
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=[lambda2])
        else:
            self.scheduler = None

    def _returns_advantages(self, rewards, dones, values, next_value):
        """
        返回每个时间步的累积折现奖励。
        :param rewards: 环境给出的奖励
        :param dones: 环境给出的 done 布尔指示符
        :param values: 价值网络给出的值
        :param next_value: 价值网络给出的下一个状态的值
        :return: 累积奖励，优势
        """

        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def training_batch(self):
        """
        执行批量训练。
        """

        start = time.time()
        reward_log = deque(maxlen=10)
        time_log = deque(maxlen=10)

        batch_size = self.config['trajectory_length']

        actions = np.empty((batch_size,), dtype=int)
        dones = np.empty((batch_size,), dtype=bool)
        rewards, values = np.empty((2, batch_size), dtype=float)
        observations = []
        observation = self.env.reset()
        observation['graph'] = observation['graph'].to(device)
        rewards_test = []
        best_reward_mean = -1000

        n_step = 0
        log_ratio = 0
        best_time = 100000
        max_improvement = 0
        while n_step < self.config['num_env_steps']:
            # Collect one batch
            probs = torch.zeros(batch_size, dtype=torch.float, device=device)
            vals = torch.zeros(batch_size, dtype=torch.float, device=device)
            probs_entropy = torch.zeros(batch_size, dtype=torch.float, device=device)

            for i in range(batch_size):
                # Collect observations from the environment
                observations.append(observation['graph'])
                policy, value = self.network(observation)
                values[i] = value.detach().cpu().numpy()
                vals[i] = value
                probs_entropy[i] = - (policy * policy.log()).sum(-1)
                try:
                    action_raw = torch.multinomial(policy, 1).detach().cpu().numpy()
                except:
                    print("chelou")
                probs[i] = policy[action_raw]
                ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
                actions[i] = -1 if action_raw == policy.shape[-1] -1 else observation['node_num'][ready_nodes][action_raw]
                observation, rewards[i], dones[i], info, improvement = self.env.step(actions[i])  # 改

                observation['graph'] = observation['graph'].to(device)
                n_step += 1

                if dones[i]:
                    observation = self.env.reset()
                    observation['graph'] = observation['graph'].to(device)
                    reward_log.append(rewards[i])
                    time_log.append(info['episode']['time'])
                    max_improvement = max(max_improvement, improvement)

            # 如果没有在最后一步结束，需要计算最后一个状态的值
            if dones[i] and not info['bad_transition']:
                next_value = 0
            else:
                next_value = self.network(observation)[1].detach().cpu().numpy()[0]

            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # 训练步骤
            loss_value, loss_actor, loss_entropy = self.optimize_model(observations, actions, probs, probs_entropy, vals, returns, advantages, step=n_step)
            if self.writer is not None and log_ratio * self.config['log_interval'] < n_step:
                log_ratio += 1
                self.writer.add_scalar('reward', np.mean(reward_log), n_step)
                self.writer.add_scalar('time', np.mean(time_log), n_step)
                self.writer.add_scalar('critic_loss', loss_value, n_step)
                self.writer.add_scalar('actor_loss', loss_actor, n_step)
                self.writer.add_scalar('entropy', loss_entropy, n_step)
                self.writer.add_scalar('improvement', max_improvement, n_step)
                if self.noise > 0:
                    current_time = np.mean([self.evaluate(), self.evaluate(), self.evaluate()])
                else:
                    current_time = self.evaluate()
                current_time = self.evaluate()
                self.writer.add_scalar('test time', current_time, n_step)
                print("comparing current time: {} with previous best: {}".format(current_time, best_time))
                if current_time < best_time:
                    print("------------saving model------------")
                    string_save = os.path.join(str(self.writer.get_logdir()), 'model{}.pth'.format(self.random_id))
                    torch.save(self.network, string_save)
                    best_time = current_time


            if len(reward_log) > 0:
                end = time.time()
                print('step ', n_step, '\n reward: ', np.mean(reward_log))

            if self.scheduler is not None:
                print(self.scheduler.get_lr())
                self.scheduler.step(int(n_step/batch_size))

        self.network = torch.load(string_save)
        results_last_model = []
        if self.noise > 0:
            for _ in range(5):
                results_last_model.append(self.evaluate())
        else:
            results_last_model.append(self.evaluate())
        torch.save(self.network, os.path.join(str(self.writer.get_logdir()), '{}_n={}_nGPU={}_nCPU={}_noise={}_time={}_improvement={}.pth'.format(self.config['env_type'], self.config['n'], self.config['nGPU'], self.config['nCPU'], self.config['noise'], str(np.mean(results_last_model)), max_improvement)))

        os.remove(string_save)
        return best_time, np.mean(results_last_model)


    def optimize_model(self, observations, actions, probs, entropies, vals, returns, advantages, step=None):
        """
        优化模型。
        :param observations: 观测值
        :param actions: 动作
        :param probs: 概率
        :param entropies: 熵
        :param vals: 价值
        :param returns: 累积奖励
        :param advantages: 优势
        :param step: 步数
        :return: 价值损失，演员损失，熵损失
        """

        returns = torch.tensor(returns[:, None], dtype=torch.float, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float, device=device)

        loss_value = 1 * F.mse_loss(vals.unsqueeze(-1), returns)
        if self.writer:
            self.writer.add_scalar('critic_loss', loss_value.data.item(), step)

        # Actor损失
        loss_policy = ((probs.log()) * advantages).mean()
        loss_entropy = entropies.mean()
        loss_actor = - loss_policy - self.entropy_cost * loss_entropy
        if self.writer:
            self.writer.add_scalar('actor_loss', loss_actor.data.item(), step)

        total_loss = self.config["loss_ratio"] * loss_value + loss_actor
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()
        return loss_value.data.item(), loss_actor.data.item(), loss_entropy.data.item()

    def evaluate(self, render=False):
        """
        评估模型。
        :param render: 是否渲染，默认为 False
        :return: 时间
        """

        env = self.monitor_env if render else deepcopy(self.env)

        observation = env.reset()
        done = False

        while not done:
            observation['graph'] = observation['graph'].to(device)
            policy, value = self.network(observation)
            action_raw = policy.argmax().detach().cpu().numpy()
            ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
            action = -1 if action_raw == policy.shape[-1] - 1 else \
                observation['node_num'][ready_nodes][action_raw].detach().numpy()[0]
            try :
                observation, reward, done, info, improvement = env.step(action)  # 改
            except KeyError:
                print("An error occurred: something is strange (chelou)!")
        return env.time

