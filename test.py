import os
import numpy as np
import time
from env import DAGEnv
import heft
import string
import matplotlib.pyplot as plt
import pandas as pd

from model import Net, SimpleNet, SimpleNet2, ResNetG, SimpleNetMax, SimpleNetW, SimpleNetWSage
import pickle as pkl
import torch
from collections import namedtuple
from torch_geometric.nn import GCNConv

if __name__ == '__main__':

    model = torch.load('runs-show/chol_n=4_nGPU=0_nCPU=4_noise=0.5_time=474.6187427062305_improvement=1.09546537338066.pth')
    model.eval()

    w_list = []
    n_node_list = []
    tile_list = []
    ready_node_list = []
    num_node_observation = []
    mean_time = []


    env_type = 'chol'
    nGPU = 0
    window = 2
    noise = 0.5
    for n in [10]:
        p_input = np.array([1] * nGPU + [0] * (4 - nGPU))
        env = DAGEnv(n, p_input, window, env_type=env_type, noise=noise)
        print(env.is_homogene)
        print("|V|: ", len(env.task_data.x))
        observation = env.reset()
        print(observation.keys())
        print(observation['graph'].x.shape)
        done = False
        time_step = 0
        total_time = 0
        improvement = 0
        while (not done) :
            start_time = time.time()
            with torch.no_grad():
                policy, value = model(observation)
            action_raw = policy.argmax().detach().cpu().numpy()
            ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
            action = -1 if action_raw == policy.shape[-1] - 1 else \
            observation['node_num'][ready_nodes][action_raw].detach().numpy()[0]

            observation, reward, done, info, improvement = env.step(action)
            cur_time = time.time() - start_time
            total_time += cur_time
            time_step += 1
            w_list.append(window)
            n_node_list.append(env.num_nodes)
            tile_list.append(n)
            mean_time.append(cur_time)

        # print('n_node:', env.num_nodes)
        print('num_nodes=', env.num_nodes, '\nFinal improvement:', improvement)


    execution_time = pd.DataFrame({'w': w_list,
                                   'n_node': n_node_list,
                                   'tiles': tile_list,
                                   'time': mean_time})
    execution_time.to_csv("img/time.csv")
