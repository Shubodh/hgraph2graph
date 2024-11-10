import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from hgraph.mol_graph_metal import MolGraphMetal
from hgraph.encoder_metal import HierMPNEncoderMetal
from hgraph.decoder_metal import HierMPNDecoderMetal
from hgraph.nnutils import *

def make_cpu(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors

class HierVAEMetal(nn.Module):

    def __init__(self,args):
        super(HierVAEMetal, self).__init__()
        self.encoder = HierMPNEncoderMetal(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depthT, args.depthG, args.dropout)
        self.decoder=HierMPNDecoderMetal(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.latent_size, args.diterT, args.diterG, args.dropout)
        self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = args.latent_size


        self.R_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.R_var = nn.Linear(args.hidden_size, args.latent_size)
    
    def rsample(self, z_vecs, W_mean, W_var, perturb=True): # this is the reparametrization trick
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss
    
    def sample(self, batch_size, greedy):
        # root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        root_vecs = torch.randn(batch_size, self.latent_size)
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)
    
    def reconstruct(self, batch):
        graphs, tensors, _ = batch
        tree_tensors, graph_tensors = tensors = make_cpu(tensors)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb=False)
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)
    
    def forward(self, graphs, tensors, orders, beta, perturb_z=True):
        tree_tensors, graph_tensors = tensors = make_cpu(tensors)

        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb_z)
        kl_div = root_kl
        # print('kl_div Inside:',kl_div)

        loss,wacc,iacc,tacc,sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc
