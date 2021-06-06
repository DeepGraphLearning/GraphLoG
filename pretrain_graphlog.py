import argparse

from loader import MoleculeDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import os, sys
import pdb
import copy
import random

from model import GNN, ProjectNet
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import ExtractSubstructureContextPair

from torch_geometric.data import DataLoader
from dataloader import DataLoaderSubstructContext

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

# Graph pooling functions
def pool_func(x, batch, mode = "mean"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)

# Mask some nodes in a graph
def mask_nodes(batch, args, num_atom_type=119):
    masked_node_indices = list()

    # select indices of masked nodes
    for i in range(batch.batch[-1] + 1):
        idx = torch.nonzero((batch.batch == i).float()).squeeze(-1)
        num_node = idx.shape[0]
        if args.mask_num == 0:
            sample_size = int(num_node * args.mask_rate + 1)
        else:
            sample_size = min(args.mask_num, int(num_node * 0.5))
        masked_node_idx = random.sample(idx.tolist(), sample_size)
        masked_node_idx.sort()
        masked_node_indices += masked_node_idx

    batch.masked_node_indices = torch.tensor(masked_node_indices)

    # mask nodes' features
    for node_idx in masked_node_indices:
        batch.x[node_idx] = torch.tensor([num_atom_type, 0])

    return batch

# NCE loss within a graph
def intra_NCE_loss(node_reps, node_modify_reps, batch, tau=0.1, epsilon=1e-6):
    node_reps_norm = torch.norm(node_reps, dim = 1).unsqueeze(-1)
    node_modify_reps_norm = torch.norm(node_modify_reps, dim = 1).unsqueeze(-1)
    sim = torch.mm(node_reps, node_modify_reps.t()) / (
        torch.mm(node_reps_norm, node_modify_reps_norm.t()) + epsilon)
    exp_sim = torch.exp(sim / tau)

    mask = torch.stack([(batch.batch == i).float() for i in batch.batch.tolist()], dim = 1)
    exp_sim_mask = exp_sim * mask
    exp_sim_all = torch.index_select(exp_sim_mask, 1, batch.masked_node_indices)
    exp_sim_positive = torch.index_select(exp_sim_all, 0, batch.masked_node_indices)
    positive_ratio = exp_sim_positive.sum(0) / (exp_sim_all.sum(0) + epsilon)

    NCE_loss = -torch.log(positive_ratio).sum() / batch.masked_node_indices.shape[0]
    mask_select = torch.index_select(mask, 1, batch.masked_node_indices)
    thr = 1. / mask_select.sum(0)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt

# NCE loss across different graphs
def inter_NCE_loss(graph_reps, graph_modify_reps, device, tau=0.1, epsilon=1e-6):
    graph_reps_norm = torch.norm(graph_reps, dim = 1).unsqueeze(-1)
    graph_modify_reps_norm = torch.norm(graph_modify_reps, dim = 1).unsqueeze(-1)
    sim = torch.mm(graph_reps, graph_modify_reps.t()) / (
        torch.mm(graph_reps_norm, graph_modify_reps_norm.t()) + epsilon)
    exp_sim = torch.exp(sim / tau)

    mask = torch.eye(graph_reps.shape[0]).to(device)
    positive = (exp_sim * mask).sum(0)
    negative = (exp_sim * (1 - mask)).sum(0)
    positive_ratio = positive / (positive + negative + epsilon)

    NCE_loss = -torch.log(positive_ratio).sum() / graph_reps.shape[0]
    thr = 1. / ((1 - mask).sum(0) + 1.)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt

# NCE loss for global-local mutual information maximization
def gl_NCE_loss(node_reps, graph_reps, batch, tau=0.1, epsilon=1e-6):
    node_reps_norm = torch.norm(node_reps, dim = 1).unsqueeze(-1)
    graph_reps_norm = torch.norm(graph_reps, dim = 1).unsqueeze(-1)
    sim = torch.mm(node_reps, graph_reps.t()) / (
            torch.mm(node_reps_norm, graph_reps_norm.t()) + epsilon)
    exp_sim = torch.exp(sim / tau)

    mask = torch.stack([(batch == i).float() for i in range(graph_reps.shape[0])], dim = 1)
    positive = exp_sim * mask
    negative = exp_sim * (1 - mask)
    positive_ratio = positive / (positive + negative.sum(0).unsqueeze(0) + epsilon)

    NCE_loss = -torch.log(positive_ratio + (1 - mask)).sum() / node_reps.shape[0]
    thr = 1. / ((1 - mask).sum(0) + 1.).unsqueeze(0)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt

# NCE loss between graphs and prototypes
def proto_NCE_loss(graph_reps, tau=0.1, epsilon=1e-6):
    global proto, proto_connection

    # similarity for original and modified graphs
    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    exp_sim_list = []
    mask_list = []
    NCE_loss = 0

    for i in range(len(proto)-1, -1, -1):
        tmp_proto = proto[i]
        proto_norm = torch.norm(tmp_proto, dim=1).unsqueeze(-1)

        sim = torch.mm(graph_reps, tmp_proto.t()) / (
                torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)
        exp_sim = torch.exp(sim / tau)

        if i != (len(proto) - 1):
            # apply the connection mask
            exp_sim_last = exp_sim_list[-1]
            idx_last = torch.argmax(exp_sim_last, dim = 1).unsqueeze(-1)
            connection = proto_connection[i]
            connection_mask = (connection.unsqueeze(0) == idx_last.float()).float()
            exp_sim = exp_sim * connection_mask

            # define NCE loss between prototypes from consecutive layers
            upper_proto = proto[i+1]
            upper_proto_norm = torch.norm(upper_proto, dim=1).unsqueeze(-1)
            proto_sim = torch.mm(tmp_proto, upper_proto.t()) / (
                    torch.mm(proto_norm, upper_proto_norm.t()) + epsilon)
            proto_exp_sim = torch.exp(proto_sim / tau)

            proto_positive_list = [proto_exp_sim[j, connection[j].long()] for j in range(proto_exp_sim.shape[0])]
            proto_positive = torch.stack(proto_positive_list, dim=0)
            proto_positive_ratio = proto_positive / (proto_exp_sim.sum(1) + epsilon)
            NCE_loss += -torch.log(proto_positive_ratio).mean()

        mask = (exp_sim == exp_sim.max(1)[0].unsqueeze(-1)).float()

        exp_sim_list.append(exp_sim)
        mask_list.append(mask)

    # define NCE loss between graph embedding and prototypes
    for i in range(len(proto)):
        exp_sim = exp_sim_list[i]
        mask = mask_list[i]

        positive = exp_sim * mask
        negative = exp_sim * (1 - mask)
        positive_ratio = positive.sum(1) / (positive.sum(1) + negative.sum(1) + epsilon)
        NCE_loss += -torch.log(positive_ratio).mean()

    return NCE_loss

# Update prototypes with batch information
def update_proto_lowest(graph_reps, decay_ratio=0.7, epsilon=1e-6):
    global proto, proto_state

    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    proto_norm = torch.norm(proto[0], dim=1).unsqueeze(-1)
    sim = torch.mm(graph_reps, proto[0].t()) / (
            torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)

    # update states of prototypes
    mask = (sim == sim.max(1)[0].unsqueeze(-1)).float()
    cnt = mask.sum(0)
    proto_state[0].data = proto_state[0].data + cnt.data

    # update prototypes
    batch_cnt = mask.t() / (cnt.unsqueeze(-1) + epsilon)
    batch_mean = torch.mm(batch_cnt, graph_reps)
    proto[0].data = proto[0].data * (cnt == 0).float().unsqueeze(-1).data + (
            proto[0].data * decay_ratio + batch_mean.data * (1 - decay_ratio)) * (cnt != 0).float().unsqueeze(-1).data

    return

# Initialze prototypes and their state
def init_proto_lowest(args, model, proj, loader, device, num_iter = 5):
    model.eval()
    proj.eval()

    for iter in range(num_iter):
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)

            # get node and graph representations
            node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
            graph_reps = pool_func(node_reps, batch.batch, mode=args.graph_pooling)

            # feature projection
            graph_reps_proj = proj(graph_reps)

            # update prototypes
            update_proto_lowest(graph_reps_proj, decay_ratio = args.decay_ratio)

    global proto, proto_state
    idx = torch.nonzero((proto_state[0] >= 2).float()).squeeze(-1)
    proto_selected = torch.index_select(proto[0], 0, idx)
    proto_selected.requires_grad = True

    return proto_selected

# Initialze prototypes and their state
def init_proto(args, index, device, num_iter = 20):
    global proto, proto_state
    proto_connection = torch.zeros(proto[index-1].shape[0]).to(device)

    for iter in range(num_iter):
        for i in range(proto[index-1].shape[0]):
            # update the closest prototype
            sim = torch.mm(proto[index], proto[index-1][i,:].unsqueeze(-1)).squeeze(-1)
            idx = torch.argmax(sim)
            if iter == (num_iter - 1):
                proto_state[index][idx] = 1
            proto_connection[i] = idx
            proto[index].data[idx, :] = proto[index].data[idx, :] * args.decay_ratio + \
                                        proto[index-1].data[i, :] * (1 - args.decay_ratio)

            # penalize rival
            sim[idx] = 0
            rival_idx = torch.argmax(sim)
            proto[index].data[rival_idx, :] = proto[index].data[rival_idx, :] * (2 - args.decay_ratio) - \
                                              proto[index-1].data[i, :] * (1 - args.decay_ratio)

    indices = torch.nonzero(proto_state[index]).squeeze(-1)
    proto_selected = torch.index_select(proto[index], 0, indices)
    proto_selected.requires_grad = True
    for i in range(indices.shape[0]):
        idx = indices[i]
        idx_connection = torch.nonzero((proto_connection == idx.float()).float()).squeeze(-1)
        proto_connection[idx_connection] = i

    return proto_selected, proto_connection

# For one epoch pretraining
def pretrain(args, model, proj, loader, optimizer, device):
    model.train()
    proj.train()

    NCE_loss_intra_cnt = 0
    NCE_loss_inter_cnt = 0
    correct_intra_cnt = 0
    correct_inter_cnt = 0
    total_intra_cnt = 0
    total_inter_cnt = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_modify = copy.deepcopy(batch)
        batch_modify = mask_nodes(batch_modify, args)
        batch, batch_modify = batch.to(device), batch_modify.to(device)

        # get node and graph representations
        node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
        node_modify_reps = model(batch_modify.x, batch_modify.edge_index, batch_modify.edge_attr)
        graph_reps = pool_func(node_reps, batch.batch, mode=args.graph_pooling)
        graph_modify_reps = pool_func(node_modify_reps, batch_modify.batch, mode=args.graph_pooling)

        # feature projection
        node_reps_proj = proj(node_reps)
        node_modify_reps_proj = proj(node_modify_reps)
        graph_reps_proj = proj(graph_reps)
        graph_modify_reps_proj = proj(graph_modify_reps)

        # NCE loss
        NCE_loss_intra, correct_intra = intra_NCE_loss(node_reps_proj, node_modify_reps_proj,
                                                                       batch_modify, tau=args.tau)
        NCE_loss_inter, correct_inter = inter_NCE_loss(graph_reps_proj, graph_modify_reps_proj,
                                                                        device, tau=args.tau)

        NCE_loss_intra_cnt += NCE_loss_intra.item()
        NCE_loss_inter_cnt += NCE_loss_inter.item()
        correct_intra_cnt += correct_intra
        correct_inter_cnt += correct_inter
        total_intra_cnt += batch_modify.masked_node_indices.shape[0]
        total_inter_cnt += graph_reps.shape[0]

        # optimization
        optimizer.zero_grad()
        NCE_loss = args.alpha * NCE_loss_intra + args.beta * NCE_loss_inter
        NCE_loss.backward()
        optimizer.step()

        if (step + 1) % args.disp_interval == 0:
            print(
                'iteration: %d, intra NCE loss: %f, intra acc: %f, inter NCE loss: %f, inter acc: %f' % (
                    step + 1, NCE_loss_intra.item(), float(correct_intra_cnt) / float(total_intra_cnt),
                    NCE_loss_inter.item(), float(correct_inter_cnt) / float(total_inter_cnt)))

    return NCE_loss_intra_cnt / step, float(correct_intra_cnt) / float(
        total_intra_cnt), NCE_loss_inter_cnt / step, float(correct_inter_cnt) / float(total_inter_cnt)

# For every epoch training
def train(args, model, proj, loader, optimizer, device):
    global proto, proto_connection
    model.train()
    proj.train()

    NCE_loss_intra_cnt = 0
    NCE_loss_inter_cnt = 0
    NCE_loss_proto_cnt = 0
    correct_intra_cnt = 0
    correct_inter_cnt = 0
    total_intra_cnt = 0
    total_inter_cnt = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_modify = copy.deepcopy(batch)
        batch_modify = mask_nodes(batch_modify, args)
        batch, batch_modify = batch.to(device), batch_modify.to(device)

        # get node and graph representations
        node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
        node_modify_reps = model(batch_modify.x, batch_modify.edge_index, batch_modify.edge_attr)
        graph_reps = pool_func(node_reps, batch.batch, mode=args.graph_pooling)
        graph_modify_reps = pool_func(node_modify_reps, batch_modify.batch, mode=args.graph_pooling)

        # feature projection
        node_reps_proj = proj(node_reps)
        node_modify_reps_proj = proj(node_modify_reps)
        graph_reps_proj = proj(graph_reps)
        graph_modify_reps_proj = proj(graph_modify_reps)

        # NCE loss
        NCE_loss_intra, correct_intra = intra_NCE_loss(node_reps_proj, node_modify_reps_proj,
                                                                       batch_modify, tau=args.tau)
        NCE_loss_inter, correct_inter = inter_NCE_loss(graph_reps_proj, graph_modify_reps_proj,
                                                                        device, tau=args.tau)
        NCE_loss_proto = proto_NCE_loss(graph_reps_proj, tau=args.tau)

        NCE_loss_intra_cnt += NCE_loss_intra.item()
        NCE_loss_inter_cnt += NCE_loss_inter.item()
        NCE_loss_proto_cnt += NCE_loss_proto.item()
        correct_intra_cnt += correct_intra
        correct_inter_cnt += correct_inter
        total_intra_cnt += batch_modify.masked_node_indices.shape[0]
        total_inter_cnt += graph_reps.shape[0]

        # optimization
        optimizer.zero_grad()
        NCE_loss = args.alpha * NCE_loss_intra + args.beta * NCE_loss_inter + \
                           args.gamma * NCE_loss_proto
        NCE_loss.backward()
        optimizer.step()

        if (step + 1) % args.disp_interval == 0:
            print(
                'iteration: %d, intra NCE loss: %f, intra acc: %f, inter NCE loss: %f, inter acc: %f' % (
                    step + 1, NCE_loss_intra.item(), float(correct_intra_cnt) / float(total_intra_cnt),
                    NCE_loss_inter.item(), float(correct_inter_cnt) / float(total_inter_cnt)))

            template = 'iteration: %d, proto NCE loss: %f'
            value_list = [step + 1, NCE_loss_proto.item()]
            for i in range(args.hierarchy):
                template += (', active num ' + str(i+1) + ': %d')
                value_list.append(proto[i].shape[0])
            print (template % tuple(value_list))

    return NCE_loss_intra_cnt / step, float(correct_intra_cnt) / float(
        total_intra_cnt), NCE_loss_inter_cnt / step, float(correct_inter_cnt) / float(
        total_inter_cnt), NCE_loss_proto_cnt / step

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GraphLoG for GNN pre-training')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='number of epochs for local learning (default: 1)')
    parser.add_argument('--global_epochs', type=int, default=10,
                        help='number of epochs for global learning (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.3,
                        help='dropout ratio (default: 0.3)')
    parser.add_argument('--mask_num', type=int, default=0,
                        help='the number of modified nodes (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max)')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataset loading')
    parser.add_argument('--tau', type=float, default=0.04, help='the temperature parameter for softmax')
    parser.add_argument('--decay_ratio', type=float, default=0.95, help='the decay ratio for moving average')
    parser.add_argument('--num_proto', type=int, default=50, help='the number of initial prototypes')
    parser.add_argument('--hierarchy', type=int, default=3, help='the number of hierarchy')
    parser.add_argument('--alpha', type=float, default=1, help='the weight of intra-graph NCE loss')
    parser.add_argument('--beta', type=float, default=1, help='the weight of inter-graph NCE loss')
    parser.add_argument('--gamma', type=float, default=0.1, help='the weight of prototype NCE loss')
    parser.add_argument('--disp_interval', type=int, default=10, help='the display interval')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("num GNN layer: %d" % (args.num_layer))

    # set up dataset and transform function.
    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up pretraining models and feature projector
    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                gnn_type=args.gnn_type).to(device)
    if args.JK == 'concat':
        proj = ProjectNet((args.num_layer + 1) * args.emb_dim).to(device)
    else:
        proj = ProjectNet(args.emb_dim).to(device)

    # set up the optimizer for pretraining
    model_param_group = [{"params": model.parameters(), "lr": args.lr},
                         {"params": proj.parameters(), "lr": args.lr}]
    optimizer_pretrain = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    # initialize prototypes and their state
    global proto, proto_state, proto_connection
    if args.JK == 'concat':
        proto = [torch.rand((args.num_proto, (args.num_layer + 1) * args.emb_dim)).to(device) for i in
                 range(args.hierarchy)]
    else:
        proto = [torch.rand((args.num_proto, args.emb_dim)).to(device) for i in range(args.hierarchy)]
    proto_state = [torch.zeros(args.num_proto).to(device) for i in range(args.hierarchy)]
    proto_connection = []

    # pre-training with only local objective
    for epoch in range(1, args.local_epochs + 1):
        print("====epoch " + str(epoch))

        train_intra_loss, train_intra_acc, train_inter_loss, train_inter_acc = pretrain(
            args, model, proj, loader, optimizer_pretrain, device)
        print(train_intra_loss, train_intra_acc, train_inter_loss, train_inter_acc)
        print("")

    # initialize prototypes and their state according to pretrained representations
    print("Initalize prototypes: layer 1")
    tmp_proto = init_proto_lowest(args, model, proj, loader, device)
    proto[0] = tmp_proto

    for i in range(1, args.hierarchy):
        print ("Initialize prototypes: layer ", i + 1)
        tmp_proto, tmp_proto_connection = init_proto(args, i, device)
        proto[i] = tmp_proto
        proto_connection.append(tmp_proto_connection)

    # set up the optimizer
    model_param_group = [{"params": model.parameters(), "lr": args.lr},
                         {"params": proj.parameters(), "lr": args.lr}]
    for i in range(args.hierarchy):
        model_param_group += [{'params': proto[i], 'lr': args.lr, 'weight_decay': 0}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    # Training with local and global objectives
    for epoch in range(1, args.global_epochs + 1):
        print("====epoch " + str(epoch))

        train_intra_loss, train_intra_acc, train_inter_loss, train_inter_acc, train_proto_loss = train(
            args, model, proj, loader, optimizer, device)
        print(train_intra_loss, train_intra_acc, train_inter_loss, train_inter_acc, train_proto_loss)

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")

    os.system('watch nvidia-smi')


if __name__ == "__main__":
    main()