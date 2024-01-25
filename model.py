import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import SAGEConv


class ModelHeterogene(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, ngcn=2, nmlp=1, nmlp_value=1, res=False, withbn=False):
        """
        初始化异构图的强化学习模型。
        :param input_dim: 输入维度
        :param hidden_dim: 隐藏层维度，默认为128
        :param ngcn: 图卷积层的数量，默认为2
        :param nmlp: MLP层的数量，默认为1
        :param nmlp_value: 用于值估计的MLP层的数量，默认为1
        :param res: 是否使用残差连接，默认为False
        :param withbn: 是否使用批归一化，默认为False
        """

        super(ModelHeterogene, self).__init__()
        self.ngcn = ngcn   # 图卷积层的数量
        self.nmlp = nmlp   # MLP 层的数量
        self.withbn = withbn   # 批归一化
        self.res = res   # 残差连接
        self.listgcn = nn.ModuleList()
        self.listmlp = nn.ModuleList()
        self.listmlp_pass = nn.ModuleList()
        self.listmlp_value = nn.ModuleList()
        self.listgcn.append(BaseConvHeterogene(input_dim, hidden_dim, 'gcn', withbn=withbn))
        for _ in range(ngcn-1):
            self.listgcn.append(BaseConvHeterogene(hidden_dim, hidden_dim, 'gcn', res=res, withbn=withbn))
        for _ in range(nmlp-1):
            self.listmlp.append(BaseConvHeterogene(hidden_dim, hidden_dim, 'mlp', res=res, withbn=withbn))
        self.listmlp.append(Linear(hidden_dim, 1))
        for _ in range(nmlp_value-1):
            self.listmlp_value.append(BaseConvHeterogene(hidden_dim, hidden_dim, 'mlp', res=res, withbn=withbn))
        self.listmlp_value.append(Linear(hidden_dim, 1))

        self.listmlp_pass.append(BaseConvHeterogene(hidden_dim+3, hidden_dim, 'mlp', withbn=withbn))
        for _ in range(nmlp-2):
            self.listmlp_pass.append(BaseConvHeterogene(hidden_dim, hidden_dim, 'mlp', res=res, withbn=withbn))
        self.listmlp_pass.append(Linear(hidden_dim, 1))

    def forward(self, dico):
        """
        前向传播函数。
        :param dico: 包含图信息、节点数量和可用节点的字典
        :return: 动作概率分布，值函数估计
        """

        data, num_node, ready = dico['graph'], dico['node_num'], dico['ready']
        x, edges = data.x, data.edge_index
        features_cluster = x[0, -3:]

        for layer in self.listgcn:
            x = layer(x, edges)

        v = torch.mean(x, dim=0)
        x_pass = torch.max(x[ready.squeeze(1).to(torch.bool)], dim=0)[0]
        x_pass = torch.cat((x_pass, features_cluster), dim=0)

        for layer in self.listmlp_value:
            v = layer(v)

        for layer in self.listmlp:
            x = layer(x)

        for layer in self.listmlp_pass:
            x_pass = layer(x_pass)

        probs = torch.cat((x[ready.squeeze(1).to(torch.bool)].squeeze(-1), x_pass), dim=0)
        probs = F.softmax(probs)
        return probs, v

class BaseConvHeterogene(torch.nn.Module):
    def __init__(self, input_dim, output_dim, type='gcn', res=False, withbn=False):
        """
        初始化异构图卷积模块。
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param type: 网络类型，'gcn' 表示使用图卷积，其他值表示使用线性层，默认为'gcn'
        :param res: 是否使用残差连接，默认为False
        :param withbn: 是否使用批归一化，默认为False
        """

        super(BaseConvHeterogene, self).__init__()
        self.res =res
        self.net_type = type
        # 根据网络类型选择是使用 GCN 还是线性层
        if type == 'gcn':
            self.layer = GCNConv(input_dim, output_dim, flow='target_to_source')
        else:
            self.layer = Linear(input_dim, output_dim)
        self.withbn = withbn
        if withbn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def forward(self, input_x, input_e=None):
        """
        前向传播函数。
        :param input_x: 输入特征
        :param input_e:  输入边信息，仅在使用图卷积时需要，默认为None
        :return: 输出特征
        """

        if self.net_type == 'gcn':
            x = self.layer(input_x, input_e)
        else:
            x = self.layer(input_x)
        # 是否启用批归一化
        if self.withbn:
            x = self.bn(x)
        # 是否启用残差连接
        if self.res:
            return F.relu(x) + input_x
        return F.relu(x)

# 以下为参考代码
class Net(torch.nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.conv_succ1 = GCNConv(input_dim, 128, flow="target_to_source")
        self.conv_succ2 = GCNConv(128, 128, flow="target_to_source")
        self.conv_succ3 = GCNConv(128, 128, flow="target_to_source")
        self.conv_pred1 = GCNConv(128, 128)

        self.conv_probs = GCNConv(128, 1, flow="target_to_source")
        self.do_nothing = Linear(128, 1)
        self.value = Linear(128, 1)

    def forward(self, dico):
        data, num_node, ready = dico['graph'], dico['node_num'], dico['ready']
        x, edge_index = data.x, data.edge_index

        x = self.conv_succ1(x, edge_index)
        x = F.relu(x)
        x = self.conv_succ2(x, edge_index)
        x = F.relu(x)
        x = self.conv_pred1(x, edge_index)
        x = F.relu(x)
        x = self.conv_succ3(x, edge_index)
        x = F.relu(x)

        probs = self.conv_probs(x, edge_index)
        x_mean = torch.mean(x, dim=0)
        v = self.value(x_mean)
        prob_nothing = self.do_nothing(x_mean)
        probs = torch.cat((probs[ready.squeeze(1).to(torch.bool)].squeeze(-1), prob_nothing), dim=0)

        probs = F.softmax(probs)

        return probs, v


class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.conv_succ1 = GCNConv(input_dim, 64, flow="target_to_source")
        # self.conv_succ2 = GCNConv(128, 128, flow="target_to_source")
        # self.conv_succ3 = GCNConv(128, 128, flow="target_to_source")
        # self.conv_pred1 = GCNConv(64, 64)

        self.conv_probs = GCNConv(64, 1, flow="target_to_source")
        self.do_nothing = Linear(64, 1)
        self.value = Linear(64, 1)

    def forward(self, dico):
        data, num_node, ready = dico['graph'], dico['node_num'], dico['ready']
        x, edge_index = data.x, data.edge_index

        x = self.conv_succ1(x, edge_index)
        x = F.relu(x)
        # x = self.conv_succ2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_pred1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_succ3(x, edge_index)
        # x = F.relu(x)

        probs = self.conv_probs(x, edge_index)
        x_mean = torch.mean(x, dim=0)
        v = self.value(x_mean)
        prob_nothing = self.do_nothing(x_mean)
        probs = torch.cat((probs[ready.squeeze(1).to(torch.bool)].squeeze(-1), prob_nothing), dim=0)

        probs = F.softmax(probs)

        return probs, v

class ResNetG(torch.nn.Module):
    def __init__(self, input_dim):
        super(ResNetG, self).__init__()
        self.conv_succ1 = GCNConv(input_dim, 64, flow="target_to_source")
        # self.conv_succ2 = GCNConv(128, 128, flow="target_to_source")
        # self.conv_succ3 = GCNConv(128, 128, flow="target_to_source")
        # self.conv_pred1 = GCNConv(64, 64)

        self.conv_probs = GCNConv(75, 1, flow="target_to_source")
        self.do_nothing = Linear(64, 1)
        self.value = Linear(64, 1)

    def forward(self, dico):
        data, num_node, ready = dico['graph'], dico['node_num'], dico['ready']
        x, edge_index = data.x, data.edge_index
        x_res = x.clone()
        x = self.conv_succ1(x, edge_index)
        x = F.relu(x)
        # x = self.conv_succ2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_pred1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_succ3(x, edge_index)
        # x = F.relu(x)

        x_input = torch.cat((x, x_res), dim=1)
        probs = self.conv_probs(x_input, edge_index)
        x_mean = torch.mean(x, dim=0)
        v = self.value(x_mean)
        prob_nothing = self.do_nothing(x_mean)
        probs = torch.cat((probs[ready.squeeze(1).to(torch.bool)].squeeze(-1), prob_nothing), dim=0)

        probs = F.softmax(probs)

        return probs, v

class SimpleNet2(torch.nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet2, self).__init__()
        self.conv_succ1 = GCNConv(input_dim, 128, flow="target_to_source")
        # self.conv_succ2 = GCNConv(128, 128, flow="target_to_source")
        # self.conv_succ3 = GCNConv(128, 128, flow="target_to_source")
        self.conv_pred1 = GCNConv(input_dim, 128)

        self.linear1 = Linear(128, 128)
        self.linear2 = Linear(128, 128)
        self.conv_probs = GCNConv(128, 1, flow="target_to_source")
        self.do_nothing = Linear(128, 1)
        self.value = Linear(128, 1)

    def forward(self, dico):
        data, num_node, ready = dico['graph'], dico['node_num'], dico['ready']
        x, edge_index = data.x, data.edge_index

        # x = self.conv_succ2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_pred1(x, edge_index)
        # x = F.relu(x)
        x = self.conv_succ1(x, edge_index)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        # x = self.conv_succ3(x, edge_index)
        # x = F.relu(x)

        probs = self.conv_probs(x, edge_index)
        x_mean = torch.mean(x, dim=0)
        v = self.value(x_mean)
        prob_nothing = self.do_nothing(x_mean)
        probs = torch.cat((probs[ready.squeeze(1).to(torch.bool)].squeeze(-1), prob_nothing), dim=0)

        probs = F.softmax(probs)

        return probs, v

class SimpleNetMax(torch.nn.Module):
    def __init__(self, input_dim):
        super(SimpleNetMax, self).__init__()
        self.conv_succ1 = GCNConv(input_dim, 64, flow="target_to_source")
        # self.conv_succ2 = GCNConv(128, 128, flow="target_to_source")
        # self.conv_succ3 = GCNConv(128, 128, flow="target_to_source")
        # self.conv_pred1 = GCNConv(64, 64)

        self.conv_probs = GCNConv(64, 1, flow="target_to_source")
        self.do_nothing = Linear(64, 1)
        self.value = Linear(64, 1)

    def forward(self, dico):
        data, num_node, ready = dico['graph'], dico['node_num'], dico['ready']
        x, edge_index = data.x, data.edge_index

        x = self.conv_succ1(x, edge_index)
        x = F.relu(x)
        # x = self.conv_succ2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_pred1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_succ3(x, edge_index)
        # x = F.relu(x)

        probs = self.conv_probs(x, edge_index)
        # x_mean = torch.mean(x, dim=0)
        x_mean = torch.max(x, dim=0)[0]
        v = self.value(x_mean)
        prob_nothing = self.do_nothing(x_mean)
        probs = torch.cat((probs[ready.squeeze(1).to(torch.bool)].squeeze(-1), prob_nothing), dim=0)

        probs = F.softmax(probs)

        return probs, v

class SimpleNetW(torch.nn.Module):
    def __init__(self, input_dim):
        super(SimpleNetW, self).__init__()
        self.conv_succ1 = GCNConv(input_dim, 64, flow="target_to_source")
        # self.conv_succ2 = GCNConv(128, 128, flow="target_to_source")
        self.conv_succ3 = GCNConv(64, 64, flow="target_to_source")
        self.conv_succ2 = GCNConv(64, 64, flow="target_to_source")
        self.conv_succ4 = GCNConv(64, 64, flow="target_to_source")
        self.conv_probs = GCNConv(64, 1, flow="target_to_source")
        self.do_nothing = Linear(64, 1)
        self.value = Linear(64, 1)

    def forward(self, dico):
        data, num_node, ready = dico['graph'], dico['node_num'], dico['ready']
        x, edge_index = data.x, data.edge_index

        x = self.conv_succ1(x, edge_index)
        x = F.relu(x)
        x = self.conv_succ2(x, edge_index)
        x = F.relu(x)
        # x = self.conv_pred1(x, edge_index)
        # x = F.relu(x)
        x = self.conv_succ3(x, edge_index)
        x = F.relu(x)
        x = self.conv_succ4(x, edge_index)
        x = F.relu(x)

        probs = self.conv_probs(x, edge_index)
        x_mean = torch.mean(x, dim=0)
        v = self.value(x_mean)
        prob_nothing = self.do_nothing(x_mean)
        probs = torch.cat((probs[ready.squeeze(1).to(torch.bool)].squeeze(-1), prob_nothing), dim=0)

        probs = F.softmax(probs)

        return probs, v

class SimpleNetWSage(torch.nn.Module):
    def __init__(self, input_dim):
        super(SimpleNetWSage, self).__init__()
        self.conv_succ1 = SAGEConv(input_dim, 64, flow="target_to_source")
        # self.conv_succ2 = GCNConv(128, 128, flow="target_to_source")
        self.conv_succ3 = SAGEConv(64, 64, flow="target_to_source")
        self.conv_succ2 = SAGEConv(64, 64, flow="target_to_source")
        self.conv_succ4 = SAGEConv(64, 64, flow="target_to_source")
        self.conv_probs = SAGEConv(64, 1, flow="target_to_source")
        self.do_nothing = Linear(64, 1)
        self.value = Linear(64, 1)