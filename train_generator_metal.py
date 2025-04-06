# python3 train_generator_metal.py --train data/metal_small_tensor --vocab data/metal_small_vocab/vocab_13.txt 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import pickle
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")
# import wandb



from hgraph import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)
# wandb.init(project="ChelateVAE_training_final", config={"learning_rate": 1e-3, "epochs": 20, "batch_size": 50})


parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab_metal)
parser.add_argument('--save_dir', required=False)
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
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
# wandb.config.update(args)
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)

with open(args.vocab) as f:
    vocab = [x.strip("\r\n ").split() for x in f]
args.vocab = PairVocabMetal(vocab,cuda=False)

model = HierVAEMetalDist(args)
# model = HierVAEMetal(args)
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

print("initialized model and ready for epochs.")


batch = pickle.load(open("data/small_3_tensor/small_3_dist.pkl", "rb"))
# print(len(batch))
meters=np.zeros(6)

for epoch in range(50):
    model.zero_grad()
    loss, kl_div, wacc, iacc, tacc, sacc, dist_loss, dist_loss_tree, count_loss = model(*batch, beta=beta, epoch_no=epoch)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    optimizer.step()
    print(f"Loss: {loss:.4f} | KL Div: {kl_div:.4f} | Wacc: {wacc:.4f} | Iacc: {iacc:.4f} | Tacc: {tacc:.4f} | Sacc: {sacc:.4f} | dist_loss: {dist_loss:.4f} | dist_loss_tree: {dist_loss_tree:.4f} | count_loss: {count_loss:.4f}")

ckpt=(model.state_dict(), optimizer.state_dict())
torch.save(ckpt, os.path.join(args.save_dir, "model.small3"))

    
# for epoch in range(args.epoch):
#     dataset = DataFolder(args.train, args.batch_size)

#     for batch in tqdm(dataset):
#         # print(len(batch))
#         total_step += 1
#         model.zero_grad()
#         loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)

#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
#         optimizer.step()

#         # we need to move the tensors from cuda to cpu first before using numpy operations on them
#         meters = meters + np.array([kl_div, loss.item(), wacc.cpu()*100, iacc.cpu()*100, tacc.cpu()*100, sacc.cpu()*100])

#         if total_step % args.print_iter == 0:
#             meters /= args.print_iter
#             print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
#             sys.stdout.flush()
#             wandb.log({
#                 "epoch": epoch,
#                 "step": total_step,
#                 "kl_divergence": meters[0],
#                 "loss": meters[1],
#                 "word_accuracy": meters[2],
#                 "input_accuracy": meters[3],
#                 "topology_accuracy": meters[4],
#                 "assembly_accuracy": meters[5],
#                 "param_norm": param_norm(model),
#                 "grad_norm": grad_norm(model),
#                 "beta": beta,
#             })
#             meters *= 0
        
#         if total_step % args.save_iter == 0:
#             ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
#             torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{total_step}"))

#         if total_step % args.anneal_iter == 0:
#             scheduler.step()
#             print("learning rate: %.6f" % scheduler.get_lr()[0])

#         if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
#             beta = min(args.max_beta, beta + args.step_beta)

# for epoch in range(args.epoch):
#     dataset=DataFolder(args.train,args.batch_size)
#     for batch in dataset:
#         total_step+=1
#         model.zero_grad()
#         kl_div = model(*batch, beta=beta) 
#         print(kl_div)

# dataset=DataFolder(args.train,args.batch_size)
# total_step=0
# beta=0

# batch = pickle.load(open("data/small_3_tensor/small_data.pkl", "rb"))
# print(len(batch))

# model.zero_grad()
# kl_div = model(*batch, beta=beta)
# print(kl_div)
# for batch in dataset:
#     print("batch")
#     print(len(batch))
#     # total_step += 1
#     # model.zero_grad()
#     # kl_div = model(*batch, beta=beta)

# print("done")
