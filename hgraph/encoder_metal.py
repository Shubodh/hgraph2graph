import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from hgraph.nnutils import *
from hgraph.mol_graph import MolGraph
from hgraph.rnn import GRU, LSTM

class MPNEncoder(nn.Module):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth, dropout):
        super(MPNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.W_o = nn.Sequential( 
                nn.Linear(node_fdim + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )

        if rnn_type == 'GRU':
            self.rnn = GRU(input_size, hidden_size, depth) 
        elif rnn_type == 'LSTM':
            self.rnn = LSTM(input_size, hidden_size, depth) 
        else:
            raise ValueError('unsupported rnn cell type ' + rnn_type)

    def forward(self, fnode, fmess, agraph, bgraph):
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
        mask[0, 0] = 0 #first node is padding
        return node_hiddens * mask, h #return only the hidden state (different from IncMPNEncoder in LSTM case)

class HierMPNEncoderMetal(nn.Module):
    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(HierMPNEncoderMetal, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.atom_size = atom_size = avocab.size()
        self.bond_size = bond_size = len(MolGraph.BOND_LIST) + 1 + MolGraph.MAX_POS # added 1 for the new bond type for iron-metal.

        self.E_c = nn.Sequential(
                nn.Embedding(vocab.size()[0], embed_size), # here vocab is an object of PairVocab so vocab.size() returns a tuple whose first element is the size of hvocab (i.e. the number of unique molecules in the dataset)
                nn.Dropout(dropout)
        )
        self.E_i = nn.Sequential(
                nn.Embedding(vocab.size()[1], embed_size),
                nn.Dropout(dropout)
        )
        self.W_c = nn.Sequential( 
                nn.Linear(embed_size + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )
        self.W_i = nn.Sequential( 
                nn.Linear(embed_size + hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(dropout)
        )

        self.E_a = torch.eye(atom_size)
        self.E_b = torch.eye( len(MolGraph.BOND_LIST)+1) # added 1 for the new bond type for iron-metal. 
        self.E_apos = torch.eye( MolGraph.MAX_POS )
        self.E_pos = torch.eye( MolGraph.MAX_POS )
        # self.E_a = torch.eye(atom_size).cuda()
        # self.E_b = torch.eye( len(MolGraph.BOND_LIST) ).cuda()
        # self.E_apos = torch.eye( MolGraph.MAX_POS ).cuda()
        # self.E_pos = torch.eye( MolGraph.MAX_POS ).cuda()

        self.W_root = nn.Sequential( 
                nn.Linear(hidden_size * 2, hidden_size), 
                nn.Tanh() #root activation is tanh
        )
        self.tree_encoder = MPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.inter_encoder = MPNEncoder(rnn_type, hidden_size + MolGraph.MAX_POS, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = MPNEncoder(rnn_type, atom_size + bond_size, atom_size, hidden_size, depthG, dropout)

    def tie_embedding(self, other):
        self.E_c, self.E_i = other.E_c, other.E_i
        self.E_a, self.E_b = other.E_a, other.E_b
    
    def embed_inter(self, tree_tensors, hatom):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        finput = self.E_i(fnode[:, 1])

        hnode = index_select_ND(hatom, 0, cgraph).sum(dim=1)
        hnode = self.W_i( torch.cat([finput, hnode], dim=-1) )

        hmess = hnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        return hnode, hmess, agraph, bgraph

    def embed_tree(self, tree_tensors, hinter):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        finput = self.E_c(fnode[:, 0])
        hnode = self.W_c( torch.cat([finput, hinter], dim=-1) )

        hmess = hnode.index_select(index=fmess[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess[:, 2])
        hmess = torch.cat( [hmess, pos_vecs], dim=-1 ) 
        return hnode, hmess, agraph, bgraph
    
    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        hnode = self.E_a.index_select(index=fnode, dim=0)
        fmess1 = hnode.index_select(index=fmess[:, 0], dim=0)
        fmess2 = self.E_b.index_select(index=fmess[:, 2], dim=0)
        fpos = self.E_apos.index_select(index=fmess[:, 3], dim=0)
        hmess = torch.cat([fmess1, fmess2, fpos], dim=-1)
        return hnode, hmess, agraph, bgraph

    def embed_root(self, hmess, tree_tensors, roots):
        roots = tree_tensors[2].new_tensor(roots) 
        fnode = tree_tensors[0].index_select(0, roots)
        agraph = tree_tensors[2].index_select(0, roots)

        nei_message = index_select_ND(hmess, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        return self.W_root(node_hiddens)

    def forward(self, tree_tensors, graph_tensors):
        tensors = self.embed_graph(graph_tensors)
        hnode,hmess,agraph,bgraph = tensors
        print(hnode.size())
        print(hmess.size())
        print(agraph.size())
        print(bgraph.size())
        hatom,_ = self.graph_encoder(*tensors)

        tensors = self.embed_inter(tree_tensors, hatom)
        hinter,_ = self.inter_encoder(*tensors)

        tensors = self.embed_tree(tree_tensors, hinter)
        hnode,hmess = self.tree_encoder(*tensors)
        hroot = self.embed_root(hmess, tensors, [st for st,le in tree_tensors[-1]])

        return hroot, hnode, hinter, hatom
    