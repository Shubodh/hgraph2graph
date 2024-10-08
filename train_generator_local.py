import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
import pickle

from torch.utils.data.dataloader import default_collate

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
import networkx as nx

from hgraph import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)


class SingleFileDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, tuple):
        return tuple(custom_collate(samples) for samples in zip(*batch))
    elif isinstance(elem, (dict, nx.DiGraph)):
        return batch
    else:
        return default_collate(batch)

parser = argparse.ArgumentParser()
# parser.add_argument('--train', required=True)
parser.add_argument('--train', required=True, help='Path to tensors-0.pkl')

parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50)
# parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

if args.load_model:
    print('continuing from checkpoint ' + args.load_model)
    model_state, optimizer_state, total_step, beta = torch.load(args.load_model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
else:
    total_step = beta = 0

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

meters = np.zeros(6)
dataset = SingleFileDataset(args.train)
# dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)

for epoch in range(args.epoch):
    # dataset = DataFolder(args.train, args.batch_size)

    # for batch in tqdm(dataset):
    for batch in tqdm(dataloader):
        total_step += 1
        model.zero_grad()
        # loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
        graphs, tensors, orders = batch
        # loss, kl_div, wacc, iacc, tacc, sacc = model(batch, beta=beta)
        loss, kl_div, wacc, iacc, tacc, sacc = model(graphs, tensors, orders, beta=beta)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        # meters = meters + np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100])
        meters = meters + np.array([kl_div, loss.item(), wacc.cpu() * 100, iacc.cpu() * 100, tacc.cpu() * 100, sacc.cpu() * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
            sys.stdout.flush()
            meters *= 0
        
        # if total_step % args.save_iter == 0:
        #     ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
        #     torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{total_step}"))

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)

    # Save model at the end of each epoch
    ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
    torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.epoch_{epoch}"))