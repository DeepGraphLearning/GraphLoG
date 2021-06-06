import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import os, sys
import numpy as np
import random

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer, scheduler):
    model.train()
    scheduler.step()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_run', type=int, default=5,
                        help='number of independent runs (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--frozen', action='store_true', default=False,
                        help='whether to freeze gnn extractor')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'bbbp', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=None, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=None, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 1, help='number of workers for dataset loading')
    args = parser.parse_args()

    if args.seed:
        seed = args.seed
        print ('Manual seed: ', seed)
    else:
        seed = random.randint(0, 10000)
        print ('Random seed: ', seed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0,
                                                                    frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,
                                                                  frac_valid=0.1, frac_test=0.1, seed=seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0,
                                                                           frac_train=0.8, frac_valid=0.1,
                                                                           frac_test=0.1, seed=seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    # run multiple times
    best_valid_auc_list = []
    last_epoch_auc_list = []
    for run_idx in range(args.num_run):
        print ('\nRun ', run_idx + 1)
        if args.runseed:
            runseed = args.runseed
            print('Manual runseed: ', runseed)
        else:
            runseed = random.randint(0, 10000)
            print('Random runseed: ', runseed)

        torch.manual_seed(runseed)
        np.random.seed(runseed)
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(runseed)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # set up model
        model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                              graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        if not args.input_model_file == "":
            model.from_pretrained(args.input_model_file)

        model.to(device)

        # set up optimizer
        # different learning rate for different part of GNN
        model_param_group = []
        if args.frozen:
            model_param_group.append({"params": model.gnn.parameters(), "lr": 0})
        else:
            model_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        scheduler = StepLR(optimizer, step_size=30, gamma=0.3)

        # run fine-tuning
        best_valid = 0
        best_valid_test = 0
        last_epoch_test = 0
        for epoch in range(1, args.epochs + 1):
            print("====epoch " + str(epoch), " lr: ", optimizer.param_groups[-1]['lr'])

            train(args, model, device, train_loader, optimizer, scheduler)

            print("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_acc = 0
            val_acc = eval(args, model, device, val_loader)
            test_acc = eval(args, model, device, test_loader)

            if val_acc > best_valid:
                best_valid = val_acc
                best_valid_test = test_acc

            if epoch == args.epochs:
                last_epoch_test = test_acc

            print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))
            print("")

        best_valid_auc_list.append(best_valid_test)
        last_epoch_auc_list.append(last_epoch_test)

    # summarize results
    best_valid_auc_list = np.array(best_valid_auc_list)
    last_epoch_auc_list = np.array(last_epoch_auc_list)

    if args.dataset in ["muv", "hiv"]:
        print('Best validation epoch:')
        print('Mean: {}\tStd: {}'.format(np.mean(best_valid_auc_list), np.std(best_valid_auc_list)))
    else:
        print('Last epoch:')
        print('Mean: {}\tStd: {}'.format(np.mean(last_epoch_auc_list), np.std(last_epoch_auc_list)))

    os.system('watch nvidia-smi')

if __name__ == "__main__":
    main()
