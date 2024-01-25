import gym
from gym.spaces import Box, Dict
import string

from env.utils import *
from env.utils import compute_graph
import heft
import matplotlib
matplotlib.use('Agg')

class DAGEnv(gym.Env):
    def __init__(self, n, node_types, window, env_type, noise=False):
        """
        初始化函数，用于创建任务调度环境。
        :param n: 任务图的节点数量
        :param node_types: 节点类型，表示任务图中每个节点的计算资源类型
        :param window: 用于构建状态表示的窗口大小
        :param env_type: 环境类型，表示任务图的类型（'LU'、'QR'、'chol'）
        :param noise: 噪声，用于添加任务执行时间的随机性
        """
        if isinstance(node_types, int):
            p = node_types
            node_types = np.ones(p)
        else:
            p = len(node_types)

        self.observation_space = Dict
        self.action_space = "Graph"

        self.noise = noise
        self.time = 0
        self.num_steps = 0
        self.p = p
        self.n = n
        self.window = window
        self.env_type = env_type
        if self.env_type == 'LU':
            self.max_duration_cpu = max(durations_cpu_lu)
            self.max_duration_gpu = max(durations_gpu_lu)
            self.task_data = ggen_denselu(self.n, self.noise)
        elif self.env_type == 'QR':
            self.max_duration_cpu = max(durations_cpu_qr)
            self.max_duration_gpu = max(durations_gpu_qr)
            self.task_data = ggen_QR(self.n, self.noise)
        elif self.env_type == 'chol':
            self.max_duration_cpu = max(durations_cpu)
            self.max_duration_gpu = max(durations_gpu)
            self.task_data = ggen_cholesky(self.n, self.noise)
        else:
            raise EnvironmentError('not implemented')
        self.num_nodes = self.task_data.num_nodes
        self.sum_task = torch.sum(self.task_data.x, dim=0)
        self.norm_desc_features = self.task_data.add_features_descendant()[0] / self.sum_task
        self.cluster = Cluster(node_types=node_types.astype(int), communication_cost=np.zeros((p, p)))
        self.running = -1 * np.ones(p)
        self.running_task2proc = {}
        self.ready_proc = np.zeros(p)
        self.ready_tasks = []
        self.processed = {}
        self.compeur_task = 0
        self.current_proc = 0
        self.is_homogene = (np.mean(self.cluster.node_types) - 1) * np.mean(self.cluster.node_types) == 0

        self.critic_path_duration = None
        self.total_work_normalized = None

        # 计算 heft 时间
        string_cluster = string.printable[:self.p]
        dic_heft = {}
        for edge in np.array(self.task_data.edge_index.t()):
            dic_heft[edge[0]] = dic_heft.get(edge[0], ()) + (edge[1],)

        def compcost(job, agent):
            idx = string_cluster.find(agent)
            expected_duration = self.task_data.task_list[job].durations[self.cluster.node_types[idx]] + self.noise * 10

            return expected_duration

        def commcost(ni, nj, A, B):
            return 0

        orders, jobson = heft.schedule(dic_heft, string_cluster, compcost, commcost)
        try:
            self.heft_time = orders[jobson[self.num_nodes - 1]][-1].end
        except:
            self.heft_time = max([v[-1] for v in orders.values() if len(v) > 0])

    def reset(self):
        """
        重置环境状态，用于开始新的任务调度。
        :return: 重新初始化后的状态表示
        """
        # self.task_data = random_ggen_fifo(self.n, self.max_in, self.max_out, self.noise)
        if self.env_type == 'LU':
            self.task_data = ggen_denselu(self.n, self.noise)
        elif self.env_type == 'QR':
            self.task_data = ggen_QR(self.n, self.noise)
        elif self.env_type == 'chol':
            self.task_data = ggen_cholesky(self.n, self.noise)
        else:
            raise EnvironmentError('not implemented')
        self.time = 0
        self.num_steps = 0
        self.running = -1 * np.ones(self.p).astype(int)
        self.running_task2proc = {}
        self.ready_proc = np.zeros(self.p)
        self.current_proc = 0

        new_ready_tasks = torch.arange(0, self.num_nodes)[
            torch.logical_not(isin(torch.arange(0, self.num_nodes), self.task_data.edge_index[1, :]))]
        self.ready_tasks = new_ready_tasks.tolist()

        self.processed = {}
        self.compeur_task = 0

        return self._compute_state()

    def step(self, action, render_before=False, render_after=False, enforce=True, speed=False):
        """
        执行一个时间步，模拟任务调度的动作。
        :param action: -1: 不执行任务；t: 在当前可用处理器上调度任务 t，仅包含 [-1, 0, ..., T] 的动作
        :param render_before: 在执行动作前是否渲染环境
        :param render_after: 在执行动作后是否渲染环境
        :param enforce: 是否强制执行动作
        :param speed: 是否以速度模式运行
        :return: 下一个状态、奖励、是否结束、信息和性能改进
        """
        self.num_steps += 1
        self._find_available_proc()

        if action == -1 and enforce:
            if len(self.running_task2proc) == 0:
                action = self.ready_tasks[0]
        if action != -1:
            self.compeur_task += 1

        self._choose_task_processor(action, self.current_proc)

        if render_before:
            self.render()
        done = self._go_to_next_action(action, enforce)
        if render_after and not speed:
            self.render()

        reward = (self.heft_time - self.time) / self.heft_time if done else 0

        if self.time != 0:
            improvement = self.heft_time / self.time
        else:
            improvement = 0

        if done:
            print("1. heft_time: ", self.heft_time, ', readys_time: ', self.time)
            print('2. reward: ', reward)
            print('3. improve (heft_time / readys_time): ', improvement)
        info = {'episode': {'r': reward, 'length': self.num_steps, 'time': self.time}, 'bad_transition': False}

        if speed:
            return 0, reward, done, info, improvement

        return self._compute_state(), reward, done, info, improvement

    def _find_available_proc(self):
        """
        寻找可用的处理器，更新当前处理器的索引。
        """
        while (self.current_proc < self.p) and (self.running[self.current_proc] > -1):
            self.current_proc += 1
        if self.current_proc == self.p:
            self.current_proc == 0
            self._forward_in_time()
        while (self.current_proc < self.p) and (self.running[self.current_proc] > -1):
            self.current_proc += 1

    def _forward_in_time(self):
        """
        推进时间，更新任务调度状态，处理完成的任务和新可用的任务。
        """
        if len(self.ready_proc[self.ready_proc > self.time]) > 0:
            min_time = np.min(self.ready_proc[self.ready_proc > self.time])
        else:
            min_time = 0

        self.time = min_time

        self.ready_proc[self.ready_proc < self.time] = self.time

        tasks_finished = self.running[np.logical_and(self.ready_proc == self.time, self.running > -1)].copy()

        self.running[self.ready_proc == self.time] = -1
        for task in tasks_finished:
            del self.running_task2proc[task]

        mask = isin(self.task_data.edge_index[0], torch.tensor(tasks_finished))
        list_succ = self.task_data.edge_index[1][mask]
        list_succ = torch.unique(list_succ)

        self.task_data.remove_edges(tasks_finished)

        new_ready_tasks = list_succ[torch.logical_not(isin(list_succ, self.task_data.edge_index[1, :]))]
        self.ready_tasks += new_ready_tasks.tolist()

        self.current_proc = np.argmin(self.running)

    def _go_to_next_action(self, previous_action, enforce=True):
        """
        根据前一个动作，决定是否推进到下一个动作，更新环境状态。
        :param previous_action: 前一个动作的值
        :param enforce: 是否强制执行动作，默认为 True
        :return: 是否完成了所有任务
        """
        has_just_passed = self.is_homogene and previous_action == -1 and enforce
        if has_just_passed:
            self._forward_in_time()
        elif previous_action == -1:
            self.current_proc += 1
        while len(self.ready_tasks) == 0:
            self._forward_in_time()
            if self._isdone():
                return True
        self._find_available_proc()
        return False

    def _choose_task_processor(self, action, processor):
        """
        选择将任务分配给处理器，并更新任务调度状态。
        :param action: 动作的值，-1 表示不执行任务
        :param processor: 处理器的索引
        """
        if action != -1:
            self.ready_proc[processor] += self.task_data.task_list[action].durations[self.cluster.node_types[processor]]
            self.ready_tasks.remove(action)
            self.processed[self.task_data.task_list[action].barcode] = [processor, self.time]
            self.running_task2proc[action] = processor
            self.running[processor] = action

    def _compute_state(self):
        """
        计算当前环境状态的表示。
        :return: 包含环境状态表示的字典，包括以下键值对：
            - 'graph': 可见图的表示。
            - 'node_num': 当前可见节点的编号。
            - 'ready': 任务是否准备好的标志。
        """
        visible_graph, node_num = compute_sub_graph(self.task_data,
                                                    torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                                 self.ready_tasks)), dtype=torch.long),
                                                    self.window)
        visible_graph.x, ready = self._compute_embeddings(node_num)
        return {'graph': visible_graph, 'node_num': node_num, 'ready': ready}

    def _remaining_time(self, running_tasks):
        """
        计算任务执行的剩余时间。
        :param running_tasks: 当前正在运行的任务列表
        :return: 包含每个任务剩余执行时间的张量
        """
        return torch.tensor(
            [self.ready_proc[self.running_task2proc[task.item()]] for task in running_tasks]) - self.time

    def _isdone(self):
        """
        判断任务调度是否完成。
        :return: 若任务调度已完成，则为True；否则为False
        """
        return (self.compeur_task == self.num_nodes and (len(self.running_task2proc) == 0))

    def _compute_embeddings(self, tasks):
        """
        计算任务的嵌入表示。
        :param tasks: 当前任务列表
        :return: 一个包含任务嵌入表示的张量和标志每个任务是否准备好的张量
        """

        ready = isin(tasks, torch.tensor(self.ready_tasks)).float()
        running = isin(tasks, torch.tensor(self.running[self.running > -1])).squeeze(-1)

        remaining_time = torch.zeros(tasks.shape[0])
        remaining_time[running] = self._remaining_time(tasks[running].squeeze(-1)).to(torch.float)
        remaining_time = remaining_time.unsqueeze(-1)

        n_succ = torch.sum((tasks == self.task_data.edge_index[0]).float(), dim=1).unsqueeze(-1)
        n_pred = torch.sum((tasks == self.task_data.edge_index[1]).float(), dim=1).unsqueeze(-1)

        task_num = self.task_data.task_list[tasks.squeeze(-1)]
        if isinstance(task_num, Task):
            task_type = torch.tensor([[4]])

        else:
            task_type = torch.tensor([task.type for task in task_num]).unsqueeze(-1)

        num_classes = 4
        one_hot_type = (task_type == torch.arange(num_classes).reshape(1, num_classes)).float()

        descendant_features_norm = self.norm_desc_features[tasks].squeeze(1)

        node_type = torch.ones(tasks.shape[0]) * self.cluster.node_types[self.current_proc]
        node_type = node_type.unsqueeze((-1))
        if sum(self.cluster.node_types == 1) == 0:
            min_ready_gpu = torch.FloatTensor([1]).repeat(tasks.shape[0]).unsqueeze((-1))
        else:
            min_ready_gpu = min(self.ready_proc[self.cluster.node_types == 1] - self.time) / self.max_duration_gpu
            min_ready_gpu = torch.FloatTensor([min_ready_gpu]).repeat(tasks.shape[0]).unsqueeze((-1))
        if sum(self.cluster.node_types == 0) == 0:
            min_ready_cpu = torch.FloatTensor([1]).repeat(tasks.shape[0]).unsqueeze((-1))
        else:
            min_ready_cpu = min(self.ready_proc[self.cluster.node_types == 0] - self.time) / self.max_duration_cpu
            min_ready_cpu = torch.FloatTensor([min_ready_cpu]).repeat(tasks.shape[0]).unsqueeze((-1))

        return (torch.cat((n_succ, n_pred, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time,
                           descendant_features_norm, node_type, min_ready_gpu, min_ready_cpu), dim=1),
                ready)

    def render(self):
        """
        渲染当前调度状态，绘制任务图和处理器通信图的可视化。
        """

        def color_task(task):
            """
            根据任务状态着色任务节点。
            """

            colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            if task in self.running:
                time_proportion = 1 - (self.ready_proc[self.running_task2proc[task]] - self.time) / \
                                  self.task_data.task_list[task].duration_cpu
                color_time = [1., time_proportion, time_proportion]
                return color_time
            elif task in self.ready_tasks:
                return colors[1]
            return colors[2]

        def color_processor(processor):
            """
            根据处理器状态着色处理器节点。
            """

            if self.running[processor] == -1:
                return [0, 1, 0] if self.current_proc == processor else [0.7, 0.7, 0.7]
            else:
                time_proportion = (self.ready_proc[processor] - self.time) / \
                                  self.task_data.task_list[self.running[processor]].duration_cpu
            return [time_proportion, 0, 0]

        visible_graph, node_num = compute_sub_graph(self.task_data,
                                                    torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                                 self.ready_tasks)), dtype=torch.long),
                                                    self.window)
        plt.figure(figsize=(8, 8))
        plt.suptitle('time: {}'.format(self.time))
        plt.subplot(121)
        plt.box(on=None)
        visible_graph.render(root=list(self.running[self.running > -1]))

        plt.subplot(122)
        plt.box(on=None)
        graph = to_networkx(Data(visible_graph.x, visible_graph.edge_index.contiguous()))
        pos = graphviz_layout(graph, prog='dot', root=None)
        node_color = [color_task(task[0].item()) for task in node_num]
        nx.draw_networkx_nodes(graph, pos, node_color=node_color)
        nx.draw_networkx_edges(graph, pos)
        labels = {}
        for i, task in enumerate(node_num):
            if task[0].item() in self.ready_tasks:
                labels[i] = task[0].item()
        nx.draw_networkx_labels(graph, pos, labels, font_size=16)
        plt.show()

        edges_list = [(u, v, {"cost": self.cluster.communication_cost[u, v]}) for u in range(self.p) for v in
                      range(self.p) if u != v]
        colors = [color_processor(p) for p in range(self.p)]
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.cluster.node_types))))
        G.add_edges_from(edges_list)
        pos = graphviz_layout(G)
        node_labels = {}
        for i, node_type in enumerate(self.cluster.node_types):
            node_labels[i] = ["CPU", "GPU"][node_type]

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=1000)
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=16)
        plt.show()

    def visualize_schedule(self, figsize=(80, 30), fig_file=None, flip=False):
        """
        可视化调度结果，生成任务调度图。
        """

        def get_data(env):
            """
            获取调度数据，包括任务处理信息。
            """

            P = env.p
            Processed = env.processed
            for k, v in Processed.items():
                Processed[k] = [int(v[0]), int(v[1])]

            makespan = int(env.time)
            data = np.ones((P, makespan)) * (-1)
            data = data.astype(int)
            compl_data = [[] for _ in range(P)]
            for x, sched in Processed.items():
                tasktype = x[0]
                pr = sched[0]
                s_time = sched[1]
                e_time = s_time + Task(x).durations[env.cluster.node_types[pr]]
                data[pr, s_time:e_time] = tasktype

                if tasktype == 0:
                    compl_data[pr].insert(0, (x[1]))
                elif tasktype == 1:
                    compl_data[pr].insert(0, (x[1], x[2]))
                elif tasktype == 2:
                    compl_data[pr].insert(0, (x[1], x[2]))
                else:
                    compl_data[pr].insert(0, (x[1], x[2], x[3]))


            return data, compl_data

        def avg(a, b):
            return (a + b) / 2.0

        P = self.p
        data, compl_data = get_data(self)
        if flip:
            data = data[-1::-1, :]
            compl_data = compl_data[-1::-1]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)

        for y, row in enumerate(data):
            x = 0
            i = 0
            indices_in_row = compl_data[y]
            while x < len(row):
                col = row[x]
                if col != -1:
                    shift = Task([col]).durations[self.cluster.node_types[y]]
                    indices = indices_in_row[i]
                else:
                    x = x + 1
                    continue
                x1 = [x, x + shift]
                y1 = np.array([y, y])
                y2 = y1 + 1
                if col == 0:
                    plt.fill_between(x1, y1, y2=y2, facecolor='green', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), 'C({})'.format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)

                if col == 1:
                    plt.fill_between(x1, y1, y2=y2, facecolor='red', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "S{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 2:
                    plt.fill_between(x1, y1, y2=y2, facecolor='orange', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "T{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 3:
                    plt.fill_between(x1, y1, y2=y2, facecolor='yellow', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "G{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                x = x + shift
                i = i + 1

        plt.ylim(P, 0)
        plt.xlim(-1e-3, data.shape[1] + 1e-3)
        plt.xticks(fontsize=50)
        if fig_file != None:
            plt.savefig(fig_file)
        return


# legacy code
class CholeskyTaskGraph(gym.Env):

    def __init__(self, n, node_types, window, noise=False):
        if isinstance(node_types, int):
            p = node_types
            node_types = np.ones(p)
        else:
            p = len(node_types)

        self.observation_space = Dict
        self.action_space = "Graph"

        self.noise = noise
        self.time = 0
        self.num_steps = 0
        self.n = n
        self.p = p
        self.window = window
        self.task_graph = compute_graph(n=n, noise=noise)
        self.task_data = TaskGraph(self.task_graph.x.clone(), self.task_graph.edge_index.clone(),
                                   self.task_graph.task_list.copy())
        self.cluster = Cluster(node_types=node_types.astype(int), communication_cost=np.zeros((p, p)))
        self.running = -1 * np.ones(p)  # array of task number
        self.running_task2proc = {}
        self.ready_proc = np.zeros(p)  # for each processor, the time where it becomes available
        self.ready_tasks = []
        self.processed = {}
        self.current_proc = 0
        self.is_homogene = (np.mean(self.cluster.node_types) - 1) * np.mean(self.cluster.node_types) == 0

        self.critic_path_duration = sum(durations_gpu[:-2]) * (self.n - 1) + durations_gpu[0]  # 158
        self.total_work_normalized = (n * durations_gpu[0] + n * (n - 1) / 2 * (durations_gpu[1] + durations_gpu[2]) + \
                                      n * (n - 1) * (n - 2) / 6 * durations_gpu[3]) / p  # 536 / p
        self.task_to_CP = np.zeros(len(self.task_graph.task_list))

    def reset(self):
        self.task_data = TaskGraph(self.task_graph.x.clone(), self.task_graph.edge_index.clone(),
                                   self.task_graph.task_list.copy())
        self.time = 0
        self.num_steps = 0
        self.running = -1 * np.ones(self.p).astype(int)
        self.running_task2proc = {}
        self.ready_proc = np.zeros(self.p)
        # self.ready_tasks.append(0)
        self.current_proc = 0

        self.ready_tasks = [0]
        self.processed = {}

        if self.noise > 0:
            for i in range(self.task_data.num_nodes):
                self.task_data.task_list[i].durations[1] = self.task_data.task_list[i].duration_gpu + np.random.normal(
                    0, self.noise)

        return self._compute_state()

    def step(self, action, render_before=False, render_after=False):
        """
        first implementation, with only [-1, 0, ..., T] actions
        :param action: -1: does nothing. t: schedules t on the current available processor
        :return: next_state, reward, done, info
        """
        self.num_steps += 1
        self._find_available_proc()
        if action == -1:
            if len(self.running_task2proc) == 0:
                # the agent does nothing but every proc is available: we enforce an arbitrary action
                action = self.ready_tasks[0]
        self._choose_task_processor(action, self.current_proc)
        if render_before:
            self.render()
        done = self._go_to_next_action(action)
        if render_after:
            self.render()
        ref = max(self.critic_path_duration, self.total_work_normalized)
        reward = - (self.time - ref) / ref if done else 0
        info = {'episode': {'r': reward, 'length': self.num_steps, 'time': self.time}, 'bad_transition': False}
        return self._compute_state(), reward, done, info

    def _find_available_proc(self):
        while (self.current_proc < self.p) and (self.running[self.current_proc] > -1):
            self.current_proc += 1
        if self.current_proc == self.p:
            # 无可用
            self.current_proc == 0
            self._forward_in_time()
        while (self.current_proc < self.p) and (self.running[self.current_proc] > -1):
            self.current_proc += 1

    def _forward_in_time(self):
        if len(self.ready_proc[self.ready_proc > self.time]) > 0:
            min_time = np.min(self.ready_proc[self.ready_proc > self.time])
        else:
            min_time = 0

        self.time = min_time

        self.ready_proc[self.ready_proc < self.time] = self.time

        tasks_finished = self.running[np.logical_and(self.ready_proc == self.time, self.running > -1)].copy()

        self.running[self.ready_proc == self.time] = -1
        for task in tasks_finished:
            del self.running_task2proc[task]

        # 计算已完成任务的后继
        mask = isin(self.task_data.edge_index[0], torch.tensor(tasks_finished))
        list_succ = self.task_data.edge_index[1][mask]
        list_succ = torch.unique(list_succ)

        # 移除节点
        self.task_data.remove_edges(tasks_finished)

        # 计算新的空闲任务
        new_ready_tasks = list_succ[torch.logical_not(isin(list_succ, self.task_data.edge_index[1, :]))]
        self.ready_tasks += new_ready_tasks.tolist()

        self.current_proc = np.argmin(self.running)

    def _go_to_next_action(self, previous_action):
        has_just_passed = self.is_homogene and previous_action == -1
        if has_just_passed:
            self._forward_in_time()
        elif previous_action == -1:
            self.current_proc += 1
        while len(self.ready_tasks) == 0:
            self._forward_in_time()
            if self._isdone():
                return True
        self._find_available_proc()
        return False

    def _choose_task_processor(self, action, processor):

        if action != -1:
            self.ready_proc[processor] += self.task_data.task_list[action].durations[self.cluster.node_types[processor]]
            self.ready_tasks.remove(action)
            self.processed[self.task_data.task_list[action].barcode] = [processor, self.time]
            self.running_task2proc[action] = processor
            self.running[processor] = action

    def _compute_state(self):
        visible_graph, node_num = compute_sub_graph(self.task_data,
                                                    torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                                 self.ready_tasks)), dtype=torch.long),
                                                    self.window)
        visible_graph.x, ready = self._compute_embeddings(node_num)
        return {'graph': visible_graph, 'node_num': node_num, 'ready': ready}

    def _remaining_time(self, running_tasks):
        return torch.tensor(
            [self.ready_proc[self.running_task2proc[task.item()]] for task in running_tasks]) - self.time

    def _isdone(self):
        return (self.task_data.edge_index.shape[-1] == 0) and (len(self.running_task2proc) == 0)

    def _compute_embeddings(self, tasks):

        ready = isin(tasks, torch.tensor(self.ready_tasks)).float()
        running = isin(tasks, torch.tensor(self.running[self.running > -1])).squeeze(-1)

        remaining_time = torch.zeros(tasks.shape[0])
        remaining_time[running] = self._remaining_time(tasks[running].squeeze(-1))
        remaining_time = remaining_time.unsqueeze(-1)

        n_succ = torch.sum((tasks == self.task_data.edge_index[0]).float(), dim=1).unsqueeze(-1)
        n_pred = torch.sum((tasks == self.task_data.edge_index[1]).float(), dim=1).unsqueeze(-1)

        task_num = self.task_data.task_list[tasks.squeeze(-1)]
        if isinstance(task_num, Task):
            task_type = torch.tensor([[4]])

        else:
            task_type = torch.tensor([task.type for task in task_num]).unsqueeze(-1)

        num_classes = 5
        one_hot_type = (task_type == torch.arange(num_classes).reshape(1, num_classes)).float()

        cpl = torch.zeros(tasks.shape[0])
        for i, task in enumerate(tasks):
            if self.task_to_CP[task] == 0:
                cpl[i] = CPAndWorkBelow(self.task_graph.task_list[task], self.n, durations_gpu)[
                             0] / self.critic_path_duration
                self.task_to_CP[task] = cpl[i]
            else:
                cpl[i] = self.task_to_CP[task]
        cpl = cpl.unsqueeze(-1)

        return (torch.cat(
            (n_succ / 10, n_pred / 10, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time / 10, cpl),
            dim=1),
                ready)


    def render(self):

        def color_task(task):
            colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            if task in self.running:
                time_proportion = 1 - (self.ready_proc[self.running_task2proc[task]] - self.time) / \
                                  self.task_data.task_list[task].duration_cpu
                color_time = [1., time_proportion, time_proportion]
                return color_time
            elif task in self.ready_tasks:
                return colors[1]
            return colors[2]

        def color_processor(processor):
            if self.running[processor] == -1:
                return [0, 1, 0] if self.current_proc == processor else [0.7, 0.7, 0.7]
            else:
                time_proportion = (self.ready_proc[processor] - self.time) / \
                                  self.task_data.task_list[self.running[processor]].duration_cpu
            return [time_proportion, 0, 0]

        visible_graph, node_num = compute_sub_graph(self.task_data,
                                                    torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                                 self.ready_tasks)), dtype=torch.long),
                                                    self.window)
        plt.figure(figsize=(8, 8))
        plt.suptitle('time: {}'.format(self.time))
        plt.subplot(121)
        plt.box(on=None)
        visible_graph.render(root=list(self.running[self.running > -1]))

        plt.subplot(122)
        plt.box(on=None)
        graph = to_networkx(Data(visible_graph.x, visible_graph.edge_index.contiguous()))
        pos = graphviz_layout(graph, prog='dot', root=None)
        node_color = [color_task(task[0].item()) for task in node_num]
        nx.draw_networkx_nodes(graph, pos, node_color=node_color)
        nx.draw_networkx_edges(graph, pos)
        labels = {}
        for i, task in enumerate(node_num):
            if task[0].item() in self.ready_tasks:
                labels[i] = task[0].item()
        nx.draw_networkx_labels(graph, pos, labels, font_size=16)
        plt.show()

        # Cluster
        edges_list = [(u, v, {"cost": self.cluster.communication_cost[u, v]}) for u in range(self.p) for v in
                      range(self.p) if u != v]
        colors = [color_processor(p) for p in range(self.p)]
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.cluster.node_types))))
        G.add_edges_from(edges_list)
        pos = graphviz_layout(G)
        node_labels = {}
        for i, node_type in enumerate(self.cluster.node_types):
            node_labels[i] = ["CPU", "GPU"][node_type]

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=1000)
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=16)
        plt.show()

    def visualize_schedule(self, figsize=(80, 30), fig_file=None, flip=False):

        def get_data(env):
            P = env.p
            Processed = env.processed
            for k, v in Processed.items():
                Processed[k] = [int(v[0]), int(v[1])]

            makespan = int(env.time)
            data = np.ones((P, makespan)) * (-1)
            data = data.astype(int)
            compl_data = [[] for _ in range(P)]
            for x, sched in Processed.items():
                tasktype = x[0]
                pr = sched[0]
                s_time = sched[1]
                e_time = s_time + Task(x).durations[env.cluster.node_types[pr]]
                data[pr, s_time:e_time] = tasktype
                if tasktype == 0:
                    compl_data[pr].insert(0, (x[1]))
                elif tasktype == 1:
                    compl_data[pr].insert(0, (x[1], x[2]))
                elif tasktype == 2:
                    compl_data[pr].insert(0, (x[1], x[2]))
                else:
                    compl_data[pr].insert(0, (x[1], x[2], x[3]))

            return data, compl_data

        def avg(a, b):
            return (a + b) / 2.0

        P = self.p
        data, compl_data = get_data(self)
        if flip:
            data = data[-1::-1, :]
            compl_data = compl_data[-1::-1]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)

        for y, row in enumerate(data):
            x = 0
            i = 0
            indices_in_row = compl_data[y]
            while x < len(row):
                col = row[x]
                if col != -1:
                    shift = Task([col]).durations[self.cluster.node_types[y]]
                    indices = indices_in_row[i]
                else:
                    x = x + 1
                    continue
                x1 = [x, x + shift]
                y1 = np.array([y, y])
                y2 = y1 + 1
                if col == 0:
                    plt.fill_between(x1, y1, y2=y2, facecolor='green', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), 'C({})'.format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)

                if col == 1:
                    plt.fill_between(x1, y1, y2=y2, facecolor='red', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "S{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 2:
                    plt.fill_between(x1, y1, y2=y2, facecolor='orange', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "T{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 3:
                    plt.fill_between(x1, y1, y2=y2, facecolor='yellow', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "G{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                x = x + shift
                i = i + 1

        plt.ylim(P, 0)
        plt.xlim(-1e-3, data.shape[1] + 1e-3)
        plt.xticks(fontsize=50)
        if fig_file != None:
            plt.savefig(fig_file)
        return


if __name__ == "__main__":
    import torch
    from env import CholeskyTaskGraph
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx

    import matplotlib.pyplot as plt
    from networkx.drawing.nx_pydot import graphviz_layout
    import numpy as np

    from model import *
