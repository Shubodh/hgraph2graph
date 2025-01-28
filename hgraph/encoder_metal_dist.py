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

    """
    1. After embed_graph - The fmess here consists of the concatenated one-hot vectors of each edge. After message passing with bgraph for the edge sharing information we get h. Now h has a size of [number of edges, hidden_size]. We index_select the hidden states of these edges corresponding to the eid stored in agraph for each node. This gives us the hidden states of the edges corresponding to each node. We sum these hidden states to get the message from the neighbours for each node. This is stored in nei_message. We concatenate the one_hot vectors of the nodes and the message from the neighbours and pass it through a linear layer to get the node_hiddens. We return the node_hiddens and the hidden states of the edges.
    """
    def forward(self, fnode, fmess, agraph, bgraph):
        # print("Inside MPNEncoderDist")
        # print(fmess.size())
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        # print(h.size())
        nei_message = index_select_ND(h, 0, agraph)
        # print("agraph ",agraph.size())
        # print("nei_message ",nei_message.size())
        nei_message = nei_message.sum(dim=1)
        # print("nei_message sum ",nei_message.size())
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
        mask[0, 0] = 0 #first node is padding
        return node_hiddens * mask, h #return only the hidden state (different from IncMPNEncoder in LSTM case)

class HierMPNEncoderMetalDist(nn.Module):
    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(HierMPNEncoderMetalDist, self).__init__()
        self.vocab = vocab
        # print("vocab.size()",vocab.size())
        # print("vocab.size()[0]",vocab.size()[0])
        # print("vocab.size()[1]",vocab.size()[1])
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.atom_size = atom_size = avocab.size()
        self.bond_size = bond_size = len(MolGraphMetal.BOND_LIST) + 1 + MolGraphMetal.MAX_POS # added 1 for the new bond type for iron-metal.

        """
        E_c - Embedding layer for the unique motifs (smiles string) from the vocab file including iron atom.
        E_i - Embedding layer of the size of all the tuples in the vocab file including iron tuple.
        W_c - 
        W_i - Linear layer followed by ReLU and dropout for the one hot vectors of the motifs and the hatom hidden states.
        """
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

        """
        These are One hot encodings (or) Identity matrices.

        E_a - atom types and its corresponding formal charges.
        E_b - bond types.
        E_apos - positions of the atoms in the molecule.
        E_pos - positions of the atoms in the molecule.
        W_root- linear layer followed by tanh activation for the root node's embedding vectors.
        """
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

        The three levels of encoder for the three encoding stages.
        """
        self.tree_encoder = MPNEncoderDist(rnn_type, hidden_size + MolGraphMetal.MAX_POS +1, hidden_size, hidden_size, depthT, dropout)
        self.inter_encoder = MPNEncoderDist(rnn_type, hidden_size + MolGraphMetal.MAX_POS +1, hidden_size, hidden_size, depthT, dropout)
        self.graph_encoder = MPNEncoderDist(rnn_type, atom_size + bond_size +1, atom_size, hidden_size, depthG, dropout)

    """
    tie_embedding is used to tie the embeddings of the encoder and decoder. This is done to reduce the number of parameters in the model. Something like sharing weights. 
    """
    def tie_embedding(self, other):
        self.E_c, self.E_i = other.E_c, other.E_i
        self.E_a, self.E_b = other.E_a, other.E_b
    
    """
    Seperating the first 4 columns of fmess and converting them to integer type and storing the distances in fdist. just concatenating the fdist to the message embeddings. Repeating the same thing for embed_tree and embed_graph.
    """

    """
    fnode - at tree level, fnode contains two columns - 0th column contains the unique index corresponding to the smiles string and the 1st column contains the unique index for each pair of (smiles,ismiles) in the tree (basically, hmap and vmap from PairVocabMetal).

    fmess_int - u, v , positional encoding in dfs traversal, 0 - u and v are the indices of the motifs in the edge.

    fdist - the distance between the two motifs in the edge.

    finput - maps the indices present in fnode[:, 1] (indices corresponding to all (smiles,ismiles) pairs) to the corresponding vector representation (embedding) in E_i.

    hnode - extracts the embedded vectors corresponding to the atoms in cgraph from hatom and sums them up for each motif in the tree. this is then concatenated with finput and passes through W_i. These are the node embeddings for the tree. 

    hmess - extracts the source node embeddings from hnode corresponding to the source node of the edge (u) from fmess_int. 

    pos_vecs - extracts the positional encoding vectors from E_pos corresponding to the positional encoding of the edges present in fmess_int.

    final hmess - concatenates the source node embeddings, positional encoding vectors and the distance between the motifs in the edge.

    """
    def embed_inter(self, tree_tensors, hatom):
        fnode, fmess, agraph, bgraph, cgraph, _ = tree_tensors
        # print("Inside embed_inter")
        # print(fnode.size())
        # print(cgraph.size())
        # print(fmess.size())
        # print("inside embed_inter")
        # print(hatom.size())
        # print(cgraph.size())

        fmess_int = fmess[:, :4].int()  # Shape: [n, 4], dtype: int32
        fdist = fmess[:, 4].unsqueeze(1)  # Shape: [n, 1]
        finput = self.E_i(fnode[:, 1])
        hnode = index_select_ND(hatom, 0, cgraph).sum(dim=1)
        hnode = self.W_i( torch.cat([finput, hnode], dim=-1) )

        hmess = hnode.index_select(index=fmess_int[:, 0], dim=0)
        pos_vecs = self.E_pos.index_select(0, fmess_int[:, 2])
        hmess = torch.cat( [hmess, pos_vecs, fdist], dim=-1 ) # added tree level edge distances to the message embeddings in hmess.
        return hnode, hmess, agraph, bgraph
    
    """
    finput - maps the indices present in fnode[:, 0] (indices corresponding to smiles string) to the corresponding vector representation (embedding) in E_c.

    hnode - concatenates the embedding vectors from finput and the hidden state vectors recieved from the MPN after embed_inter.

    hmess - extracts vectors from hnode corresponding to the source node of the edge (u) from fmess_int.
    pos_vecs - extracts the positional encoding vectors from E_pos corresponding to the positional encoding of the edges present in fmess_int.

    final hmess - concatenates the source node embeddings, positional encoding vectors and the distance between the motifs in the edge.

    """

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
    
    """
    embed_graph creates the first embeddings at the graph level and returns it.
    fmess_int is the first 4 columns of fmess converted to integer type and fdist is the 5th column of fmess.

    fnode - At graph level, fnode contains the unique index for each pair of (atom, formalcharge) in the graph.
    fmess_int - u, v ,bondtype, child_order(from assm_candidates) - u and v are the indices of the atoms in the edge.
    fdist - the distance between the two atoms

    hnode - extracts one-hot vectors from E_a corresponding to the atoms in fnode from E_a.

    fmess1 - extracts the one-hot vectors corresponding to the source atom of the edge (u) from hnode which contains the one-hot vectors of all the atoms in the graph.

    fmess2 - extracts the one-hot vectors from E_b corresponding to the bond_type of the edges present in fmess_int

    fpos - extracts the one-hot vectors from E_apos corresponding to the child_order of the edges present in fmess_int

    hmess - concatenates the one-hot vectors of the source atom, bond type, child_order and the distance between the atoms in the edge.
    """

    """
    IDEAS GIVEN DURING MEET BY PRANAV AND BISWAJIT SIR
    ### using fdist as a bin maybe. 
    ### combining one hot with fdist may not be semantically well. can put mlp after concatenating. will make it compatible. popular in papers. 
    """
    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        # print("inside embed_graph")
        # print(fnode.size())
        # print(fnode)
        # print(fnode.size())
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
        # print(fmess1.shape)
        # print(fmess2.shape)
        # print(fpos.shape)
        # print("hmess")
        # print(hmess.size())
        return hnode, hmess, agraph, bgraph

    """
    roots - tensor of the root nodes of the tree.

    fnode - extracts the embedded vectors of the nodes from embed_tree corresponding to the root nodes.

    agraph - extracts the vectors from agraph corresponding to the root nodes.

    nei_message - extracts the vectors from hmess corresponding to the root nodes's neighbours and sums it. So this now contains the aggregated messages from the neighbours of the root nodes.

    node_hiddens - concatenates the embedded vectors of the root nodes and the aggregated messages from the neighbours of the root nodes.

    returns the final hidden states of the root nodes after passing through a linear layer.
    """
    def embed_root(self, hmess, tree_tensors, roots):
        # print("inside embed_root")
        # print(roots)
        roots = tree_tensors[2].new_tensor(roots) 
        # print(roots.shape)
        fnode = tree_tensors[0].index_select(0, roots)
        agraph = tree_tensors[2].index_select(0, roots)

        nei_message = index_select_ND(hmess, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        return self.W_root(node_hiddens)

    def forward(self, tree_tensors, graph_tensors):
        """
        The tensors are in the order of the following: - 
        
        1. tree_tensors - fnode, fmess, agraph, bgraph, cgraph, tree_scope
        2. graph_tensors - fnode, fmess, agraph, bgraph, graph_scope
        """
        # print("Inside HierMPNEncoderMetalDist")
        # print("fmess - tree_tensors[1] : ",tree_tensors[1].size())
        # print("agraph - tree_tensors[2] : ",tree_tensors[2].size())
        # print("bgraph - tree_tensors[3] : ",tree_tensors[3].size())
        tensors = self.embed_graph(graph_tensors)
        # hnode,hmess,agraph,bgraph = tensors
        # print(hnode.size())
        # print(hmess.size())
        # print(agraph.size())
        # print(bgraph.size())
        # print("embed graph success")
        """
        tensors contains - hnode, hmess, agraph, bgraph
        this is sent to the MPN Encoder and the hatom is basically the final hidden states for each node after messsage aggregation and then transformation using a linear+relu+dropout layer.
        """
        hatom,_ = self.graph_encoder(*tensors)
        # print("graph encoder success")
        # print(hatom.size())
        # print(len(tree_tensors))
        """
        -> The final hidden states from graph_encoder along with the tree_tensors are sent to embed_inter. 
        -> tensors consists of the embedded vectors at the motif level for the tree.
        """
        tensors = self.embed_inter(tree_tensors, hatom)
        # print("embed inter success")
        """
        hinter consists of the the final hidden states for each motif after message aggregation from the edges and transformation using a linear+relu+dropout layer.
        """
        hinter,_ = self.inter_encoder(*tensors)
        # print("inter encoder success")

        tensors = self.embed_tree(tree_tensors, hinter)
        # print("embed tree success")
        hnode,hmess = self.tree_encoder(*tensors)

        """
        here st is the offsets of the start of each graph.
        """
        hroot = self.embed_root(hmess, tensors, [st for st,le in tree_tensors[-1]])
        """
        Returns each level of embeddings for the tree, motif and graph along with the root. 
        """
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
        """
        fnode, fmess, agraph, bgraph, cgraph (for tree) are selected based on the subset of nodes and edges present in the subgraph and returned.
        """
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
        """
        num_tree_nodes - number of nodes in the tree (motifs)
        num_graph_nodes - number of nodes in the graph (atoms)

        subgraph[0] - node ids in the subgraph
        subgraph[1] - edge ids in the subgraph

        """
        num_tree_nodes = tree_tensors[0].size(0)
        num_graph_nodes = graph_tensors[0].size(0)

        if len(subgraph[0]) + len(subgraph[1]) > 0:
            # print("Subgraph ",subgraph)
            sub_graph_tensors = self.get_sub_tensor(graph_tensors, subgraph)[:-1] #graph tensor is already embedded
            # print(sub_graph_tensors)
            hgraph.node, hgraph.mess = self.graph_encoder(sub_graph_tensors, hgraph.mess, num_graph_nodes, subgraph)
            # print(hgraph.mess)

        if len(subtree[0]) + len(subtree[1]) > 0:
            sub_inter_tensors = self.embed_sub_tree(inter_tensors, hgraph.node, subtree, is_inter_layer=True)
            hinter.node, hinter.mess = self.inter_encoder(sub_inter_tensors, hinter.mess, num_tree_nodes, subtree)

            sub_tree_tensors = self.embed_sub_tree(tree_tensors, hinter.node, subtree, is_inter_layer=False)
            htree.node, htree.mess = self.tree_encoder(sub_tree_tensors, htree.mess, num_tree_nodes, subtree)

        return htree, hinter, hgraph
    
class IncHierMPNEncoderMetalDist_DistancePrediction(HierMPNEncoderMetalDist):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout):
        super(IncHierMPNEncoderMetalDist_DistancePrediction, self).__init__(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        """
        Added +1 for the new distance dimension in the embeddings.
        """
        self.graph_encoder = IncMPNEncoderDist(rnn_type, self.atom_size + self.bond_size + 1, self.atom_size, hidden_size, depthG, dropout)
        del self.W_root

    def get_sub_tensor(self, tensors, subset):
        """
        fnode, fmess, agraph, bgraph, cgraph (for tree) are selected based on the subset of nodes and edges present in the subgraph and returned.
        """
        subnode, submess = subset
        fnode, fmess, agraph, bgraph = tensors[:4]
        fnode, fmess = fnode.index_select(0, subnode), fmess.index_select(0, submess)
        agraph, bgraph = agraph.index_select(0, subnode), bgraph.index_select(0, submess)

        if len(tensors) == 6:
            cgraph = tensors[4].index_select(0, subnode)
            return fnode, fmess, agraph, bgraph, cgraph, tensors[-1]
        else:
            return fnode, fmess, agraph, bgraph, tensors[-1]

    def forward(self, graph_tensors, hgraph, subgraph):
        """
        num_tree_nodes - number of nodes in the tree (motifs)
        num_graph_nodes - number of nodes in the graph (atoms)

        subgraph[0] - node ids in the subgraph
        subgraph[1] - edge ids in the subgraph

        """
        num_graph_nodes = graph_tensors[0].size(0)

        if len(subgraph[0]) + len(subgraph[1]) > 0:
            # print("Subgraph ",subgraph)
            sub_graph_tensors = self.get_sub_tensor(graph_tensors, subgraph)[:-1] #graph tensor is already embedded
            # print(sub_graph_tensors)
            hgraph.node, hgraph.mess = self.graph_encoder(sub_graph_tensors, hgraph.mess, num_graph_nodes, subgraph)
            # print(hgraph.mess)

        return hgraph

    
