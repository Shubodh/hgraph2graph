import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from hgraph.nnutils import *
from hgraph.mol_graph_metal import MolGraphMetal
from hgraph.rnn import GRU, LSTM

class MPNEncoderDist(nn.Module):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth, dropout):
        super(MPNEncoderDist, self).__init__()
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

class HierMPNEncoderMetalDist(nn.Module):
    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(HierMPNEncoderMetalDist, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.atom_size = atom_size = avocab.size()
        self.bond_size = bond_size = len(MolGraphMetal.BOND_LIST) + 1 + MolGraphMetal.MAX_POS # added 1 for the new bond type for iron-metal.

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
        self.E_b = torch.eye( len(MolGraphMetal.BOND_LIST)+1) # added 1 for the new bond type for iron-metal. 
        self.E_apos = torch.eye( MolGraphMetal.MAX_POS )
        self.E_pos = torch.eye( MolGraphMetal.MAX_POS )
        # self.E_a = torch.eye(atom_size).cuda()
        # self.E_b = torch.eye( len(MolGraphMetal.BOND_LIST) ).cuda()
        # self.E_apos = torch.eye( MolGraphMetal.MAX_POS ).cuda()
        # self.E_pos = torch.eye( MolGraphMetal.MAX_POS ).cuda()

        self.W_root = nn.Sequential( 
                nn.Linear(hidden_size * 2, hidden_size), 
                nn.Tanh() #root activation is tanh
        )
        """
        Added +1 for the new distance dimension in the embeddings at all three levels of the heirarchical structure.
        """
        self.tree_encoder = MPNEncoderDist(rnn_type, hidden_size + MolGraphMetal.MAX_POS +1, hidden_size, hidden_size, depthT, dropout)
        self.inter_encoder = MPNEncoderDist(rnn_type, hidden_size + MolGraphMetal.MAX_POS +1, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = MPNEncoderDist(rnn_type, atom_size + bond_size +1, atom_size, hidden_size, depthG, dropout)

    def tie_embedding(self, other):
        self.E_c, self.E_i = other.E_c, other.E_i
        self.E_a, self.E_b = other.E_a, other.E_b
    
    """
    Seperating the first 4 columns of fmess and converting them to integer type and storing the distances in fdist. just concatenating the fdist to the message embeddings. Repeating the same thing for embed_tree and embed_graph.
    """
    def embed_inter(self, tree_tensors, hatom):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        # print("Inside embed_inter")
        # print(fnode.size())
        # print(fmess.size())

        fmess_int = fmess[:, :4].int()  # Shape: [n, 4], dtype: int32
        fdist = fmess[:, 4].unsqueeze(1)  # Shape: [n, 1]
        finput = self.E_i(fnode[:, 1])

        hnode = index_select_ND(hatom, 0, cgraph).sum(dim=1)
        hnode = self.W_i( torch.cat([finput, hnode], dim=-1) )

        hmess = hnode.index_select(index=fmess_int[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess_int[:, 2])
        hmess = torch.cat( [hmess, pos_vecs, fdist], dim=-1 ) # added tree level edge distances to the message embeddings in hmess.
        return hnode, hmess, agraph, bgraph

    def embed_tree(self, tree_tensors, hinter):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors

        fmess_int = fmess[:, :4].int()  # Shape: [n, 4], dtype: int32
        fdist = fmess[:, 4].unsqueeze(1)  # Shape: [n, 1]

        finput = self.E_c(fnode[:, 0])
        hnode = self.W_c( torch.cat([finput, hinter], dim=-1) )

        hmess = hnode.index_select(index=fmess_int[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess_int[:, 2])
        hmess = torch.cat( [hmess, pos_vecs, fdist], dim=-1 ) 
        return hnode, hmess, agraph, bgraph
    
    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        # print("datatype of fmess")
        # print(fmess.dtype)
        # print(fmess[6])
        # print("size of fmess")
        # print(fmess.shape)
        fmess_int = fmess[:, :4].int()  # Shape: [n, 4], dtype: int32
        fdist = fmess[:, 4].unsqueeze(1)  # Shape: [n, 1]
        # print("fmess_int")
        # print(fmess_int.dtype)
        # print(type(fmess_int))

        hnode = self.E_a.index_select(index=fnode, dim=0)
        fmess1 = hnode.index_select(index=fmess_int[:, 0], dim=0)
        fmess2 = self.E_b.index_select(index=fmess_int[:, 2], dim=0)
        fpos = self.E_apos.index_select(index=fmess_int[:, 3], dim=0)
        hmess = torch.cat([fmess1, fmess2, fpos, fdist], dim=-1) # Modified to include fdist
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
        # hnode,hmess,agraph,bgraph = tensors
        # print(hnode.size())
        # print(hmess.size())
        # print(agraph.size())
        # print(bgraph.size())
        # print("embed graph success")
        hatom,_ = self.graph_encoder(*tensors)
        # print("graph encoder success")
        # print(hatom.size())
        # print(len(tree_tensors))
        tensors = self.embed_inter(tree_tensors, hatom)
        # print("embed inter success")
        hinter,_ = self.inter_encoder(*tensors)
        # print("inter encoder success")

        tensors = self.embed_tree(tree_tensors, hinter)
        # print("embed tree success")
        hnode,hmess = self.tree_encoder(*tensors)
        # print("tree encoder success")
        hroot = self.embed_root(hmess, tensors, [st for st,le in tree_tensors[-1]])
        # print("embed root success")

        return hroot, hnode, hinter, hatom
    

class IncMPNEncoderDist(MPNEncoderDist):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth, dropout):
        super(IncMPNEncoderDist, self).__init__(rnn_type, input_size, node_fdim, hidden_size, depth, dropout)

    def forward(self, tensors, h, num_nodes, subset):
        fnode, fmess, agraph, bgraph = tensors
        subnode, submess = subset

        if len(submess) > 0: 
            h = self.rnn.sparse_forward(h, fmess, submess, bgraph)

        nei_message = index_select_ND(self.rnn.get_hidden_state(h), 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
        node_hiddens = index_scatter(node_hiddens, node_buf, subnode)
        return node_hiddens, h

class IncHierMPNEncoderMetalDist(HierMPNEncoderMetalDist):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(IncHierMPNEncoderMetalDist, self).__init__(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        """
        Added +1 for the new distance dimension in the embeddings.
        """
        self.tree_encoder = IncMPNEncoderDist(rnn_type, hidden_size + MolGraphMetal.MAX_POS + 1, hidden_size, hidden_size, depthT, dropout)
        self.inter_encoder = IncMPNEncoderDist(rnn_type, hidden_size + MolGraphMetal.MAX_POS + 1, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = IncMPNEncoderDist(rnn_type, self.atom_size + self.bond_size + 1, self.atom_size, hidden_size, depthG, dropout)
        del self.W_root

    def get_sub_tensor(self, tensors, subset):
        subnode, submess = subset
        fnode, fmess, agraph, bgraph = tensors[:4]
        fnode, fmess = fnode.index_select(0, subnode), fmess.index_select(0, submess)
        agraph, bgraph = agraph.index_select(0, subnode), bgraph.index_select(0, submess)

        if len(tensors) == 6:
            cgraph = tensors[4].index_select(0, subnode)
            return fnode, fmess, agraph, bgraph, cgraph, tensors[-1]
        else:
            return fnode, fmess, agraph, bgraph, tensors[-1]

    def embed_sub_tree(self, tree_tensors, hinput, subtree, is_inter_layer):
        subnode, submess = subtree
        num_nodes = tree_tensors[0].size(0)
        fnode, fmess, agraph, bgraph, cgraph, _ = self.get_sub_tensor(tree_tensors, subtree)

        if is_inter_layer:
            finput = self.E_i(fnode[:, 1])
            hinput = index_select_ND(hinput, 0, cgraph).sum(dim=1)
            hnode = self.W_i( torch.cat([finput, hinput], dim=-1) )
        else:
            finput = self.E_c(fnode[:, 0])
            hinput = hinput.index_select(0, subnode)
            hnode = self.W_c( torch.cat([finput, hinput], dim=-1) )

        if len(submess) == 0:
            hmess = fmess
        else:
            """
            Modified to include the distances in the embeddings.
            """
            node_buf = torch.zeros(num_nodes, self.hidden_size, device=fmess.device)
            node_buf = index_scatter(hnode, node_buf, subnode)
            fmess_int = fmess[:, :4].int()  # Shape: [n, 4], dtype: int32
            fdist = fmess[:, 4].unsqueeze(1)  # Shape: [n, 1]
            hmess = node_buf.index_select(index=fmess_int[:, 0], dim=0)
            pos_vecs = self.E_pos.index_select(0, fmess_int[:, 2])
            hmess = torch.cat( [hmess, pos_vecs, fdist], dim=-1 ) # added tree level edge distances to the message embeddings in hmess.
        return hnode, hmess, agraph, bgraph 

    def forward(self, tree_tensors, inter_tensors, graph_tensors, htree, hinter, hgraph, subtree, subgraph):
        num_tree_nodes = tree_tensors[0].size(0)
        num_graph_nodes = graph_tensors[0].size(0)

        if len(subgraph[0]) + len(subgraph[1]) > 0:
            sub_graph_tensors = self.get_sub_tensor(graph_tensors, subgraph)[:-1] #graph tensor is already embedded
            hgraph.node, hgraph.mess = self.graph_encoder(sub_graph_tensors, hgraph.mess, num_graph_nodes, subgraph)

        if len(subtree[0]) + len(subtree[1]) > 0:
            sub_inter_tensors = self.embed_sub_tree(inter_tensors, hgraph.node, subtree, is_inter_layer=True)
            hinter.node, hinter.mess = self.inter_encoder(sub_inter_tensors, hinter.mess, num_tree_nodes, subtree)

            sub_tree_tensors = self.embed_sub_tree(tree_tensors, hinter.node, subtree, is_inter_layer=False)
            htree.node, htree.mess = self.tree_encoder(sub_tree_tensors, htree.mess, num_tree_nodes, subtree)

        return htree, hinter, hgraph
    
