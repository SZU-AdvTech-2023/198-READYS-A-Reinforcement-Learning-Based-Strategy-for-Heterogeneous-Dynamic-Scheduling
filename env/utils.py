from torch_geometric.data import Data

import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx

import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import re
import os
import pickle as pkl
# 任务类型 0: POTRF 1:SYRK 2:TRSM 3: GEMMS

# 异构
durations_cpu = [18, 57, 52, 95, 0]
durations_gpu = [11, 2, 8, 3, 0]

durations_cpu_lu = [18, 95, 52, 52]
durations_gpu_lu = [11, 3, 8, 8]

durations_cpu_qr = [4, 6, 6, 10]
durations_gpu_qr = [3, 3, 1, 1]

# 同构
# durations_cpu = [1, 3, 3, 6, 0]
# durations_gpu = [11, 2, 8, 3, 0]

# durations_cpu = [24, 52, 57, 95, 0]
# durations_gpu2 = [12, 1, 3, 2, 0]

simple_durations = [1, 3, 3, 6, 0]

colors = {0: [0, 0, 0], 1: [230, 190, 255], 2: [170, 255, 195], 3: [255, 250, 200],
          4: [255, 216, 177], 5: [250, 190, 190], 6: [240, 50, 230], 7: [145, 30, 180], 8: [67, 99, 216],
          9: [66, 212, 244], 10: [60, 180, 75], 11: [191, 239, 69], 12: [255, 255, 25], 13: [245, 130, 49],
          14: [230, 25, 75], 15: [128, 0, 0], 16: [154, 99, 36], 17: [128, 128, 0], 18: [70, 153, 144],
          19: [0, 0, 117]}
color_normalized = {i: list(np.array(colors[i])/255) for i in colors}


class Task():
    def __init__(self, barcode, noise=0, task_type='chol'):
        """
        初始化任务。
        :param barcode: 任务条形码，包含任务类型等信息
        :param noise: 噪声水平，默认为0
        :param task_type: 任务类型，'chol' 表示 Cholesky 分解，'LU' 表示 LU 分解，'QR' 表示 QR 分解，默认为'chol'
        """

        self.type = barcode[0]
        if task_type=='chol':
            self.duration_cpu = durations_cpu[self.type]
            self.duration_gpu = durations_gpu[self.type]

        elif task_type == 'LU':
            self.duration_cpu = durations_cpu_lu[self.type]
            self.duration_gpu = durations_gpu_lu[self.type]

        elif task_type == 'QR':
            self.duration_cpu = durations_cpu_qr[self.type]
            self.duration_gpu = durations_gpu_qr[self.type]
        else:
            raise NotImplementedError('task type unknown')
        self.barcode = barcode
        self.durations = [self.duration_cpu, self.duration_gpu]
        # if noise and self.type == 3:
        #     if np.random.uniform() < 1/15:
        #         self.durations[-1] *= 3
        if noise > 0:
            self.durations[0] += np.max([np.random.normal(0, noise), -2])


class TaskGraph(Data):
    def __init__(self, x, edge_index, task_list):
        """
        初始化任务数据。
        :param x: 节点特征张量
        :param edge_index: 边索引张量
        :param task_list: 任务列表
        """

        Data.__init__(self, x, edge_index.to(torch.long))
        self.task_list = np.array(task_list)
        self.task_to_num = {v: k for (k, v) in enumerate(self.task_list)}
        self.n = len(self.x)

    def render(self, root=None):
        """
        绘制任务数据的可视化图。
        :param root: 图的根节点，默认为 None
        """

        task_list = [t.barcode for t in self.task_list]
        graph = to_networkx(Data(self.x, self.edge_index.contiguous()))
        pos = graphviz_layout(graph, prog='dot', root=root)
        node_color = [color_normalized[task[0]] for task in task_list]
        nx.draw_networkx_nodes(graph, pos, node_color=node_color)
        nx.draw_networkx_edges(graph, pos)

    def remove_edges(self, node_list):
        """
        移除指定节点的边。
        :param node_list: 需要移除边的节点列表
        """

        mask_edge = isin(self.edge_index[0, :], torch.tensor(node_list)) | \
                    isin(self.edge_index[1, :], torch.tensor(node_list))
        self.edge_index = self.edge_index[:, torch.logical_not(mask_edge)]

    def add_features_descendant(self):
        """
        为任务数据添加后代特征。
        :return: 包含添加后代特征后的两个张量，分别为 `succ_features_norm` 和 `succ_features`
        """

        n = self.n
        x = self.x
        succ_features = torch.zeros((n, 4))
        succ_features_norm = torch.zeros((n, 4))
        edges = self.edge_index
        for i in reversed(range(n)):
            succ_i = edges[1][edges[0] == i]
            feat_i = x[i] + torch.sum(succ_features[succ_i], dim=0)
            n_pred_i = torch.FloatTensor([torch.sum(edges[1] == j) for j in succ_i])
            if len(n_pred_i) == 0:
                feat_i_norm = x[i]
            else:
                feat_i_norm = x[i] + torch.sum(succ_features_norm[succ_i] / n_pred_i.unsqueeze(1).repeat((1, 4)), dim=0)
            succ_features[i] = feat_i
            succ_features_norm[i] = feat_i_norm
        return succ_features_norm, succ_features
        # return succ_features_norm, succ_features/succ_features[0]

class Node():
    def __init__(self, type):
        self.type = type

class Cluster():
    def __init__(self, node_types, communication_cost):
        self.node_types = node_types
        self.node_state = np.zeros(len(node_types))
        self.communication_cost = communication_cost


    def render(self):
        """
        渲染函数。
        :param self: 类的实例，其中包含了图结构信息（如节点类型、通信成本等）
        :return: 无返回值，但该函数绘制并显示了图的可视化表示
        """
        edges_list = [(u, v, {"cost": w}) for (u, v, w) in enumerate(self.communication_cost)]
        colors = ["k" if node_type == 0 else "red" for node_type in self.node_types]
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.node_types))))
        G.add_edges_from(edges_list)
        pos = graphviz_layout(G)

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors)
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos)


def succASAP(task, n, noise):
    """
    根据给定的任务和节点数量，计算该任务的后续任务集合。
    :param task: 一个表示当前任务的对象，其中包含任务类型和条形码信息
    :param n: 整数参数，表示系统中的总节点数
    :param noise: 布尔值或噪声参数，用于决定是否在生成后续任务时引入随机性
    :return: 包含所有可能后续任务的列表，每个任务都是 Task 类型的对象
    """
    tasktype = task.type
    i = task.barcode[1]
    j = task.barcode[2]
    k = task.barcode[3]
    listsucc = []
    if tasktype == 0:
        if i < n:
            for j in range(i + 1, n + 1, 1):
                y = (2, i, j, 0)
                listsucc.append(Task(y, noise))
        else:
            y = (4, 0, 0, 0)
            listsucc.append(Task(y))

    if tasktype == 1:
        if j < i - 1:
            y = (1, i, j + 1, 0)
            listsucc.append(Task(y))
        else:
            y = (0, i, 0, 0)
            listsucc.append(Task(y, noise))

    if tasktype == 2:
        if i <= n - 1:
            for k in range(i + 1, j):
                y = (3, k, j, i)
                listsucc.append(Task(y, noise))
            for k in range(j + 1, n + 1):
                y = (3, j, k, i)
                listsucc.append(Task(y, noise))
            y = (1, j, i, 0)
            listsucc.append(Task(y, noise))

    if tasktype == 3:
        if k < i - 1:
            y = (3, i, j, k + 1)
            listsucc.append(Task(y, noise))
        else:
            y = (2, i, j, 0)
            listsucc.append(Task(y, noise))

    return listsucc


def CPAndWorkBelow(x, n, durations):
    """
    计算给定任务在其完成之前所需的最早完成时间（CPl）以及总工作量（TotalWork）。
    :param x: 表示当前任务的对象，其 `barcode` 属性存储了任务的相关属性
    :param n: 系统中的总节点数
    :param durations: 包含不同阶段持续时间的列表 [C, S, T, G]
    :return: 一个包含两个元素的元组：(CPl, TotalWork)
    """
    x_bar = x.barcode
    C = durations[0]
    S = durations[1]
    T = durations[2]
    G = durations[3]
    ReadyTasks = []
    ReadyTasks.append(x_bar)
    Seen = []
    ToVisit = []
    ToVisit.append(x_bar)
    TotalWork = durations[x_bar[0]]
    CPl = 0

    tasktype = x_bar[0]
    if tasktype == 0:
        CPl = C + (n - x_bar[1]) * (T + S + C)
    if tasktype == 1:
        CPl = (x_bar[1] - x_bar[2]) * S + C + (n - x_bar[1]) * (T + S + C)
    if tasktype == 2:
        CPl = (x_bar[2] - x_bar[1] - 1) * (T + G) + (n - x_bar[2] + 1) * (T + S + C)
    if tasktype == 3:
        CPl = (x_bar[1] - x_bar[3]) * G + (x_bar[2] - x_bar[1] - 1) * (T + G) + (n - x_bar[2] + 1) * (T + S + C)

    return (CPl, TotalWork)

def _add_task(dic_already_seen, list_to_process, task):
    """
    将任务添加到已访问列表及待处理列表中，同时避免重复添加。
    """
    if task.barcode in dic_already_seen:
        pass
    else:
        dic_already_seen[task.barcode] = len(dic_already_seen)
        list_to_process.append(task)


def _add_node(dic_already_seen, list_to_process, node):
    """
    将节点添加到已访问列表及待处理列表中，同时避免重复添加。
    """
    if node in dic_already_seen:
        pass
    else:
        dic_already_seen[node] = True
        list_to_process.append(node)


def compute_graph(n, noise=False):
    """
    根据任务类型、节点数量和可选的噪声参数，递归地构建并返回一个基于任务转换关系的任务图数据结构（TaskGraph）。
    :param n: 节点总数
    :param noise: 是否在计算任务转换时引入随机性，默认为 False
    :return: 一个 TaskGraph 对象，包含了节点特征向量、边索引以及任务列表
    """
    root_nodes = []
    TaskList = {}
    EdgeList = []

    root_nodes.append(Task((0, 1, 0, 0), noise))
    TaskList[(0, 1, 0, 0)] = 0

    while len(root_nodes) > 0:
        task = root_nodes.pop()
        list_succ = succASAP(task, n, noise)
        for t_succ in list_succ:
            _add_task(TaskList, root_nodes, t_succ)
            EdgeList.append((TaskList[task.barcode], TaskList[t_succ.barcode]))

    embeddings = [k for k in TaskList]

    data = Data(x=torch.tensor(embeddings, dtype=torch.float),
                edge_index=torch.tensor(EdgeList).t().contiguous())

    task_array = []
    for (k, v) in TaskList.items():
        task_array.append(Task(k, noise=noise))
    return TaskGraph(x=torch.tensor(embeddings, dtype=torch.float),
                edge_index=torch.tensor(EdgeList).t().contiguous(), task_list=task_array)


def isin(ar1, ar2):
    """
    检查一个数组 ar1 中是否存在与另一个数组 ar2 中任意元素相同的元素。
    :param ar1: 第一个输入数组
    :param ar2: 第二个输入数组
    :return: 如果 ar1 中存在 ar2 的元素，则返回 True，否则返回 False
    """
    return (ar1[..., None] == ar2).any(-1)

def remove_nodes(edge_index, mask, num_nodes):
    """
    移除图中根据给定掩码标记的节点，并相应更新边索引。
    :param edge_index: 边索引张量
    :param mask: 一个布尔型掩码，指示需要移除的节点
    :param num_nodes: 图中的原始节点总数
    :return: 更新后的边索引张量，排除了被移除节点相关的边
    """
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=mask.device)
    assoc[mask] = torch.arange(mask.sum(), device=assoc.device)
    edge_index = assoc[edge_index]

    return edge_index

def compute_sub_graph(data, root_nodes, window):
    """
    计算以指定根节点集合为基础，在给定窗口大小内的子图。
    :param data: 输入的原始任务图数据结构（TaskGraph）
    :param root_nodes: 作为子图起始点的节点索引列表
    :param window: 搜索子图时的最大宽度（步数）
    :return: 子任务图（TaskGraph）及保留节点在原图中的位置
    """
    already_seen = torch.zeros(data.num_nodes, dtype=torch.bool)
    already_seen[root_nodes] = 1
    edge_list = torch.tensor([[], []], dtype=torch.long)

    i = 0
    while len(root_nodes) > 0 and i < window:
        mask = isin(data.edge_index[0], root_nodes)
        list_succ = data.edge_index[1][mask]
        list_pred = data.edge_index[0][mask]

        edge_list = torch.cat((edge_list, torch.stack((list_pred, list_succ))), dim=1)

        list_succ = torch.unique(list_succ)

        list_succ = list_succ[already_seen[list_succ] == 0]
        already_seen[list_succ] = 1
        root_nodes = list_succ
        i += 1

    assoc = torch.full((len(data.x),), -1, dtype=torch.long)
    assoc[already_seen] = torch.arange(already_seen.sum())

    node_num = torch.nonzero(already_seen)
    new_x = data.x[already_seen]
    new_edge_index = remove_nodes(data.edge_index, already_seen, len(data.x))
    mask_edge = (new_edge_index != -1).all(dim=0)
    new_edge_index = new_edge_index[:, mask_edge]
    new_task_list = data.task_list[already_seen]

    return TaskGraph(new_x, new_edge_index, new_task_list), node_num

def taskGraph2SLC(taskGraph, save_path):
    """
    将给定的任务图转化为特定格式并保存到文件中。
    :param taskGraph: 要转换的任务图数据结构
    :param save_path: 输出文件路径
    :return: 无返回值，直接写入文件
    """
    with open(save_path,"w") as file:
        file.write(str(len(taskGraph.task_list)))
        file.write('\n')
        for node, task in enumerate(taskGraph.task_list):
            line1 = str(node + 1) + " " + str(simple_durations[task.type]) + " 1"
            file.write(line1)
            file.write('\n')

            line2 = ""
            for n in taskGraph.edge_index[1][taskGraph.edge_index[0] == node]:
                line2 += str(n.item() + 1) + " 0 "
            line2 += "-1"
            file.write(line2)
            file.write('\n')

def random_ggen_fifo_edges(n_vertex, max_in, max_out):
    """
    使用 ggen 工具生成具有先进先出 (FIFO) 特性的随机有向图的边信息。
    :param n_vertex: 图中的顶点数量
    :param max_in: 每个顶点最大入度数
    :param max_out: 每个顶点最大出度数
    :return: 边索引数组，表示图中的边关系
    """
    stream = os.popen("ggen generate-graph fifo {:d} {:d} {:d}".format(n_vertex, max_in, max_out))
    graph = stream.read()
    out_graph = graph.split('dag')[1].replace('\n', '').replace('\t', '').replace('{', '[[').replace('}', ']]')
    out_graph = out_graph.replace(' -> ', ', ')
    out_graph = out_graph.replace(';', '], [')
    out_graph = eval(out_graph)
    out_graph.pop()
    edge_index = np.transpose(np.array(out_graph))
    return edge_index


def random_ggen_fifo(n_vertex, max_in, max_out, noise=0):
    """
    根据 FIFO 约束生成随机任务图，并将其转换为 TaskGraph 数据结构。
    :param n_vertex: 图中的节点数量
    :param max_in: 每个节点的最大输入边数
    :param max_out: 每个节点的最大输出边数
    :param noise: 是否引入噪声（默认为0，即不引入）
    :return: 随机生成的 TaskGraph 结构
    """
    edges = random_ggen_fifo_edges(n_vertex, max_in, max_out)
    n = np.max(edges) + 1
    tasks = np.random.randint(0, 4, size=n)
    x = np.zeros((n, 4), dtype=int)
    x[np.arange(n), tasks] = 1
    task_list = []
    for t in tasks:
        task_list.append(Task((t), noise=noise))
    return TaskGraph(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edges), task_list=task_list)

def ggen_cholesky(n_vertex, noise=0):
    """
    使用 ggen 工具生成 Cholesky 分解算法对应的有向图，并转换为 TaskGraph 结构。
    :param n_vertex: 图中的顶点数量
    :param noise: 是否添加噪音（默认为0，即不添加）
    :return: 表示 Cholesky 分解任务图的 TaskGraph 对象
    """
    dic_task_ch = {'p': 0, 's': 1, 't': 2, 'g': 3}
    def parcours_and_purge(s):
        reg = re.compile('\[kernel=[\D]*\]')
        x = reg.findall(s)
        return np.array([dic_task_ch[subx[8:9]] for subx in x])
    def parcours_and_purge_edges(s):
        reg = re.compile('\\t[\d]+ -> [\d]+\\t')
        x = reg.findall(s)
        out = np.array([[int(subx.split(' -> ')[0][1:]), int(subx.split(' -> ')[1][:-1])] for subx in x])
        return out.transpose()
    file_path = 'graphs/cholesky_{}.txt'.format(n_vertex)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            graph = f.read()
    else:
        stream = os.popen("ggen dataflow-graph cholesky {:d}".format(n_vertex))
        graph = stream.read()
    edges = parcours_and_purge_edges(graph)
    tasks = parcours_and_purge(graph)
    n = len(tasks)
    x = np.zeros((n, 4), dtype=int)
    x[np.arange(n), tasks.astype(int)] = 1
    task_list = []
    for i, t in enumerate(tasks):
        task_list.append(Task((t, i), noise=noise, task_type='chol'))
    return TaskGraph(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edges), task_list=task_list)

def ggen_denselu(n_vertex, noise=0):
    """
    使用 ggen 工具生成基于 DenseLU 算法的数据流图，并转换为 TaskGraph 结构。
    :param n_vertex: 图中的顶点数量
    :param noise: 是否添加噪音（默认为0，即不添加）
    :return: 表示 DenseLU 任务图的 TaskGraph 对象
    """
    dic_task_lu = {'lu': 0, 'bm': 1, 'bd': 2, 'fw': 3}

    def parcours_and_purge(s):
        reg = re.compile('\[kernel=[\D]*\]')
        x = reg.findall(s)
        return np.array([dic_task_lu[subx[8:10]] for subx in x])

    def parcours_and_purge_edges(s):
        reg = re.compile('\\t[\d]+ -> [\d]+\\t')
        x = reg.findall(s)
        out = np.array([[int(subx.split(' -> ')[0][1:]), int(subx.split(' -> ')[1][:-1])] for subx in x])
        return out.transpose()

    file_path = 'graphs/denselu_{}.txt'.format(n_vertex)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            graph = f.read()
    else:
        stream = os.popen("ggen dataflow-graph denselu {:d}".format(n_vertex))
        graph = stream.read()
    edges = parcours_and_purge_edges(graph)
    tasks = parcours_and_purge(graph)
    n = len(tasks)
    x = np.zeros((n, 4), dtype=int)
    x[np.arange(n), tasks] = 1
    task_list = []
    for i, t in enumerate(tasks):
        task_list.append(Task((t, i), noise=noise, task_type='LU'))
    return TaskGraph(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edges), task_list=task_list)


def ggen_QR(n, noise=0):
    """
    手动构造 QR 分解算法对应的有向图，并转换为 TaskGraph 结构。
    :param n: 图中的矩阵维度，也代表了节点的数量
    :param noise: 是否添加噪音（默认为0，即不添加）
    :return: 表示 QR 分解任务图的 TaskGraph 对象
    """
    file_path = 'graphs/QR_{}.pkl'.format(n)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            output = pkl.load(f)
            return output
        print('file loaded')

    numtask = 0

    numtasks = {}
    listtask = []

    for i in range(1, n + 1):
        numtasks[0, i, 0, 0] = numtask
        numtask = numtask + 1
        listtask.append(0)

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            numtasks[1, j, i, 0] = numtask
            numtask = numtask + 1
            listtask.append(1)

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            numtasks[2, i, j, 0] = numtask
            numtask = numtask + 1
            listtask.append(2)

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            for k in range(i + 1, n + 1):
                numtasks[3, i, j, k] = numtask
                numtask = numtask + 1
                listtask.append(3)

    listedges = []
    for i in range(1, n):
        source = numtasks[0, i, 0, 0]
        for j in range(i + 1, n + 1):
            dest = numtasks[1, j, i, 0]
            listedges.append((source, dest))

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            source = numtasks[1, j, i, 0]
            for k in range(i + 1, n + 1):
                dest = numtasks[3, i, j, k]
                listedges.append((source, dest))

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            source = numtasks[2, i, j, 0]
            for k in range(i + 1, n + 1):
                dest = numtasks[3, i, k, j]
                listedges.append((source, dest))

    for i in range(1, n - 1):
        for j in range(i + 2, n + 1):
            for k in range(i + 2, n + 1):
                source = numtasks[3, i, j, k]
                dest = numtasks[3, i + 1, j, k]
                listedges.append((source, dest))

    for i in range(1, n - 1):
        for k in range(i + 2, n):
            source = numtasks[3, i, i + 1, k]
            dest = numtasks[2, i + 1, k, 0]
            listedges.append((source, dest))

    for i in range(1, n - 1):
        for j in range(i + 2, n):
            source = numtasks[3, i, j, i + 1]
            dest = numtasks[1, j, i + 1, 0]
            listedges.append((source, dest))

    for i in range(1, n - 1):
        source = numtasks[3, i, i + 1, i + 1]
        dest = numtasks[0, i + 1, 0, 0]
        listedges.append((source, dest))

    tasks = np.array(listtask)
    edges = np.array(listedges).transpose()

    n = len(tasks)
    x = np.zeros((n, 4), dtype=int)
    x[np.arange(n), tasks] = 1
    task_list = []
    for i, t in enumerate(tasks):
        task_list.append(Task((t, i), noise=noise, task_type='QR'))
    return TaskGraph(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edges), task_list=task_list)

