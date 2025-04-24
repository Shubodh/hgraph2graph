import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from hgraph.nnutils import *
from hgraph.encoder_metal_dist import IncHierMPNEncoderMetalDist, IncHierMPNEncoderMetalDist_DistancePrediction
from hgraph.mol_graph_metal import MolGraphMetal
from hgraph.inc_graph_metal import IncTreeMetal, IncGraphMetal
from collections import defaultdict

class HTuple():
    def __init__(self, node=None, mess=None, vmask=None, emask=None):
        self.node, self.mess = node, mess
        self.vmask, self.emask = vmask, emask

class HierMPNDecoderMetalDist(nn.Module):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, latent_size, depthT, depthG, dropout, attention=False):
        super(HierMPNDecoderMetalDist, self).__init__()
        """
        self.hmpn - IncHierMPNEncoderMetalDist is the sub class of the original HierMPNEncoderMetalDist class for the decoder. Check encoder_metal_dist.py for notes on this subclass. 

        self.E_assm - E_i is the Embedding layer of the size of all the tuples in the vocab file including iron tuple. it is copied from the encoder. previously in hgnn_metal.py they use tie_embedding to share weights between the encoders E_i and decoder E_i.

        self.E_order - E_order defines the one-hot vectors for each position of the atom in the molecule. MAX_POS is predefined to be 20. NEED TO INCREASE THIS FOR LARGER MOLECULES WHEN EXPANDING THE DATASET.

        self.topoNN - Takes the concatenated hidden and latent vectors as input and outputs a scalar value. The output might be used for binary classification.

        self.clsNN - Takes the concatenated hidden and latent vectors as input and outputs a vector of logits of size equal to the number of unique motifs (smiles strings - 0th column) in the vocab file.

        self.iclsNN - Takes the concatenated hidden and latent vectors as input and outputs a vector of logits of size equal to the number of tuples (smiles & ismiles strings - 1st column) in the vocab file.

        self.matchNN - 
        
        self.W_assm - a linear layer that projects the hidden size to the latent space. 
        self.W_root - a linear layer that projects the latent size to the hidden space, this is used at the start to convert the src_root_vecs to the init_vecs. 
        self.W_dist - a linear layer that projects the hidden size to 1 for predicting distances.

        ATTENTION MECHANISM: (False by default)

            self.A_topo, self.A_cls, self.A_assm - 3 linear layers that project the hidden size to the latent size.

        LOSS FUNCTIONS:

            self.topo_loss - Binary Cross Entropy Loss for binary classification on the output of topoNN.
            self.cls_loss - Cross Entropy Loss multiclass classification task on the output of clsNN.
            self.icls_loss - Cross Entropy Loss multiclass classification task on the output of iclsNN.
            self.assm_loss - Cross Entropy Loss multiclass classification task 
        
        """
        self.vocab = vocab
        self.avocab = avocab
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.use_attention = attention
        # self.itensor = torch.LongTensor([]).cuda()
        self.itensor = torch.LongTensor([])

        self.hmpn = IncHierMPNEncoderMetalDist(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        self.hmpn_distances=IncHierMPNEncoderMetalDist_DistancePrediction(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        self.hmpn_motifdist=IncHierMPNEncoderMetalDist(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        self.rnn_cell = self.hmpn.tree_encoder.rnn
        self.rnn_cell_atom = self.hmpn_distances.graph_encoder.rnn # for atom level distance predictions
        self.rnn_cell_motif =self.hmpn_motifdist.tree_encoder.rnn # for motif level distance predictions
        self.E_assm = self.hmpn.E_i 
        # self.E_order = torch.eye(MolGraphMetal.MAX_POS).cuda()
        self.E_order = torch.eye(MolGraphMetal.MAX_POS)

        """
        Handles topological predictions, e.g., determining connectivity or structural relationships between nodes.
        Input: Concatenated hidden and latent vectors.
        Output: A scalar value (e.g., for binary classification).
        """
        self.topoNN = nn.Sequential(
                nn.Linear(hidden_size + latent_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
        )
        self.clsNN = nn.Sequential(
                nn.Linear(hidden_size + latent_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, vocab.size()[0])
        )
        self.iclsNN = nn.Sequential(
                nn.Linear(hidden_size + latent_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, vocab.size()[1])
        )
        self.matchNN = nn.Sequential(
                nn.Linear(hidden_size + embed_size + MolGraphMetal.MAX_POS, hidden_size),
                nn.ReLU(),
        )
        self.W_assm = nn.Linear(hidden_size, latent_size)

        # self.W_dist = nn.Sequential(
        #         nn.Linear(hidden_size, 2 * hidden_size),  # First layer (expansion)
        #         nn.ReLU(),

        #         nn.Linear(2 * hidden_size, 4*hidden_size),  # Second layer (compression)
        #         nn.ReLU(),

        #         nn.Linear(4*hidden_size, 8*hidden_size),
        #         nn.ReLU(),

        #         nn.Linear(8*hidden_size, 4*hidden_size),
        #         nn.ReLU(),

        #         nn.Linear(4*hidden_size, 2*hidden_size),
        #         nn.ReLU(),

        #         nn.Linear(2 * hidden_size, hidden_size),  # Second layer (compression)
        #         nn.ReLU(),

        #         nn.Linear(hidden_size, 1)  # Output layer (distance prediction)
        # )
        self.W_dist = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size),
                nn.ReLU(),

                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),

                nn.Linear(hidden_size, 1)
        )
        self.W_dist_motifs = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size),
                nn.ReLU(),

                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),

                nn.Linear(hidden_size, 1) 
        )

        # self.regressor = nn.Sequential(
        #     nn.Linear(latent_size, latent_size*2),
        #     nn.ReLU(),
        #     nn.Linear(latent_size*2, latent_size),
        #     nn.ReLU(),
        #     nn.Linear(latent_size, 1)
        # )

        self.ligand_classifier = nn.Sequential(
            nn.Linear(latent_size, latent_size*2),
            nn.ReLU(),
            nn.Linear(latent_size*2, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 10)
        )

        if latent_size != hidden_size:
            self.W_root = nn.Linear(latent_size, hidden_size)

        if self.use_attention:
            self.A_topo = nn.Linear(hidden_size, latent_size)
            self.A_cls = nn.Linear(hidden_size, latent_size)
            self.A_assm = nn.Linear(hidden_size, latent_size)

        self.topo_loss = nn.BCEWithLogitsLoss(size_average=False)
        self.cls_loss = nn.CrossEntropyLoss(size_average=False)
        self.icls_loss = nn.CrossEntropyLoss(size_average=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)
        self.dist_loss = nn.MSELoss(size_average=False)
        self.dist_loss_motifs = nn.MSELoss(size_average=False)
        # self.regression_loss = nn.MSELoss(size_average=False)
        self.classification_loss = nn.CrossEntropyLoss(size_average=False)
        
    def apply_tree_mask(self, tensors, cur, prev):
        """
        Just the usual masking of tensors.

        cur.emask: Indicates which edges (or connections) in the current graph are active.
        prev.vmask: Indicates which vertices (or nodes) in the previous graph are active.
        """
        fnode, fmess, agraph, bgraph, cgraph, scope = tensors
        agraph = agraph * index_select_ND(cur.emask, 0, agraph)
        bgraph = bgraph * index_select_ND(cur.emask, 0, bgraph)
        cgraph = cgraph * index_select_ND(prev.vmask, 0, cgraph)
        return fnode, fmess, agraph, bgraph, cgraph, scope

    def apply_graph_mask(self, tensors, hgraph):
        fnode, fmess, agraph, bgraph, scope = tensors
        agraph = agraph * index_select_ND(hgraph.emask, 0, agraph)
        bgraph = bgraph * index_select_ND(hgraph.emask, 0, bgraph)
        return fnode, fmess, agraph, bgraph, scope

    def update_graph_mask(self, graph_batch, new_atoms, hgraph):
        """
        hgraph is a Htuple which has mess, vmask, emask attributes initialized. vmask is a tensor of zeros of size len(fnode) and emask is a tensor of zeros of size len(fmess).

        new_atom_index - its a tensor of the same dtpye and device as vmask made of the new_atoms list. 

        hgraph.vmask.scatter_ - this is an in-place operation that assigns 1 to the indices of the new_atoms in the vmask tensor. Basically, it activates the nodes (new_atoms) by marking their indices in the vmask tensor.

        new_atom_set - set of the new_atoms list.

        for loop:
            iterate over the new_atoms list. for each atom check its neighbours if they are present in the new_atoms_set. If present, append the bond's index to the new_bonds list.
        
        new_bond_index - tensor of the same dtype and device as emask made of the new_bonds (mess_idx) list.

        hgraph.emask.scatter - this is an in-place operation that assigns 1 to the indices of the new_bonds in the emask tensor. Basically, it identifies edges (bonds) connecting new_atoms and marks their indices in the emask tensor.

        returns the new_atom_index and new_bond_index.

        Node Mask Update (vmask): Activates nodes (new_atoms) by marking their indices in the vmask tensor.
        Edge Mask Update (emask): Identifies edges (bonds) connecting new_atoms and marks their indices in the emask tensor.
        """
        new_atom_index = hgraph.vmask.new_tensor(new_atoms)
        hgraph.vmask.scatter_(0, new_atom_index, 1)

        new_atom_set = set(new_atoms)
        new_bonds = [] #new bonds are the subgraph induced by new_atoms
        for zid in new_atoms:
            for nid in graph_batch[zid]:
                if nid not in new_atom_set: continue
                new_bonds.append( graph_batch[zid][nid]['mess_idx'] )
        
        # #print("new_bonds:",new_bonds)

        new_bond_index = hgraph.emask.new_tensor(new_bonds)
        if len(new_bonds) > 0:
            hgraph.emask.scatter_(0, new_bond_index, 1)
        return new_atom_index, new_bond_index

 
    def init_decoder_state(self, tree_batch, tree_tensors, src_root_vecs):

    # !need to make changes - check caps in brackets for places with problems 
        """
        init_decoder_state

        Input: tree_batch, tree_tensors, src_root_vecs (or init_vecs - hidden state size)

        batch_size - length of init_vecs (hidden layer size along dim 1) - number of root nodes in the batch. 
        num_mess - number of messages (fmess) in the batch at the tree (or motif) level.
        agraph, bgraph - neighbouring nodes and edges information of the batch of metal complexes.

        Running a loop over the scope of the tree_tensors of the batch:
            - tup[0] - root node index of the tree, tup[1] - length of the complex starting from the tree node.

            - the agraph entry corresponding to the root node is updated to num_mess + i. The root nodes are associated with unique identifiers that depend on their position in the current batch. 
            
            # !(MIGHT NEED TO CHANGE) : This can be done in their approach because the tree structure for them is like a top-down tree with the root node being the first node in the tree and having only one neighbour. In our case, the root node (IRON) has multiple neighbors. This might not have the last column of the agraph entry to be 0 and have a unique identifier over there so need to maybe add one more column.
            #^ (IT WORKS ON THIS SMALL DATASET)

            - mess_idx is the unique edge id associated with each edge in the batch. The update to bgraph with num_mess + i assigns a unique identifier to the message at each step, incrementing num_mess and indexing it by i to ensure a unique identifier for the message. Basically, for all the edges between the root node and its neighbours, the last column of the bgraph entry of these edges is updated to num_mess + i. This keeps track of the initial edges between the root node and its neighbours.

            #! (MIGHT NEED TO CHANGE) : if the last column of the bgraph entry of the root-neighbour pair need not be empty everytime. 
            #^ (IT WORKS ON THIS SMALL DATASET)
        
        htree - HTuple object with mess and emask attributes initialized while node and vmask are None.
            
            - htree.mess - it is a tuple of two tensors. 0th tensor is initialized with the concatenation of zeros whose size is (len(fmess), hidden_size) and the src_root_vecs. 1st vector is a tensor of zeros whose size is (len(fmess) + len(src_root+vecs), hidden_size). 
            
            - htree.emask - It is a one-dimensional tensor made of zeros (length equal to fmess) concatenated with ones (length equal to batch_size i.e. number of root nodes). This is used to mask the messages in the batch because now the bgraph of these messages which sprout from the root node are marked with num_mess+i and the last entries of emask corresponding to 1 also have the same indices.


        (DOUBT) - one required change might be to change the scope in MolGraphMetal since it signifies the rootnode of a graph. currently it is set to each ligands start and not to the complex's start of Fe atom. 
        #*(THIS CHANGE IS DONE)

        """
        batch_size = len(src_root_vecs)
        num_mess = len(tree_tensors[1])
        agraph = tree_tensors[2].clone()
        bgraph = tree_tensors[3].clone()

        for i,tup in enumerate(tree_tensors[-1]):
            root = tup[0]
            assert agraph[root,-1].item() == 0
            agraph[root,-1] = num_mess + i
            for v in tree_batch.successors(root):
                mess_idx = tree_batch[root][v]['mess_idx'] 

                assert bgraph[mess_idx,-1].item() == 0
                bgraph[mess_idx,-1] = num_mess + i

        new_tree_tensors = tree_tensors[:2] + [agraph, bgraph] + tree_tensors[4:]
        new_tree_tensors_motif = tree_tensors[:2] + [agraph, bgraph] + tree_tensors[4:]
        htree = HTuple()
        htree_motif = HTuple()
        htree.mess = self.rnn_cell.get_init_state(tree_tensors[1], src_root_vecs)
        htree_motif.mess = self.rnn_cell.get_init_state(tree_tensors[1], src_root_vecs)
        htree.emask = torch.cat( [bgraph.new_zeros(num_mess), bgraph.new_ones(batch_size)], dim=0 )
        htree_motif.emask = torch.cat([bgraph.new_zeros(num_mess), bgraph.new_ones(batch_size)], dim=0)

        return htree, htree_motif, new_tree_tensors, new_tree_tensors_motif

    def attention(self, src_vecs, batch_idx, queries, W_att):
        size = batch_idx.size()
        if batch_idx.dim() > 1:
            batch_idx = batch_idx.view(-1)
            queries = queries.view(-1, queries.size(-1))

        src_vecs = src_vecs.index_select(0, batch_idx)
        att_score = torch.bmm( src_vecs, W_att(queries).unsqueeze(-1) )
        att_vecs = F.softmax(att_score, dim=1) * src_vecs
        att_vecs = att_vecs.sum(dim=1)
        return att_vecs if len(size) == 1 else att_vecs.view(size[0], size[1], -1)

    def get_topo_score(self, src_tree_vecs, batch_idx, topo_vecs):
        if self.use_attention:
            topo_cxt = self.attention(src_tree_vecs, batch_idx, topo_vecs, self.A_topo)
        else:
            topo_cxt = src_tree_vecs.index_select(index=batch_idx, dim=0)
        return self.topoNN( torch.cat([topo_vecs, topo_cxt], dim=-1) ).squeeze(-1)

    def get_cls_score(self, src_tree_vecs, batch_idx, cls_vecs, cls_labs):
        """
        It takes the cluster vectors and cluster labels as input and returns the cluster scores and icluster scores using the neural networks clsNN and iclsNN respectively.
        """
        if self.use_attention:
            cls_cxt = self.attention(src_tree_vecs, batch_idx, cls_vecs, self.A_cls)
        else:
            cls_cxt = src_tree_vecs.index_select(index=batch_idx, dim=0)

        cls_vecs = torch.cat([cls_vecs, cls_cxt], dim=-1)
        cls_scores = self.clsNN(cls_vecs)

        if cls_labs is None: #inference mode
            icls_scores = self.iclsNN(cls_vecs) #no masking
        else:
            vocab_masks = self.vocab.get_mask(cls_labs)
            icls_scores = self.iclsNN(cls_vecs) #apply mask by log(x + mask): mask=0 or -INF
        return cls_scores, icls_scores

    def get_assm_score(self, src_graph_vecs, batch_idx, assm_vecs):
        if self.use_attention:
            assm_cxt = self.attention(src_graph_vecs, batch_idx, assm_vecs, self.A_assm)
        else:
            assm_cxt = index_select_ND(src_graph_vecs, 0, batch_idx)
        return (self.W_assm(assm_vecs) * assm_cxt).sum(dim=-1)

    def forward(self, src_mol_vecs, graphs, tensors, orders, ligand_counts, epoch_no):

        """
        batch_size - len(orders) is the length of all_orders, which is the number of root nodes in the tree/graph batch (or number of metal complexes.)
        tree_batch - Tree structure of the batch of metal complexes in a continuous manner using offsets.
        graph_batch - Graph structure of the batch of metal complexes in a continuous manner using offsets.

        tensors - Tuple of tree_tensors and graph_tensors.
        inter_tensors - tree_tensors is used as inter_tensors.

        src_mol_vecs - Tuple of src_root_vecs, src_tree_vecs, src_graph_vecs. All three of these are the same z_vecs from the reparametrization trick which draws a sample from the latent space using the mean and variance vectors obtained from the root_vector which the encoder returns.  

        init_vecs = Uses a linear layer to project the src_root_vecs to the hidden size if the latent size is not equal to the hidden size.

        htree, tree_tensors:
            Input: tree_batch, tree_tensors, init_vecs. The tree_batch and tree_tensors here are the same ones which were made in tensorize function.
            Output : Modified tree tensors with updated agraph and bgraph with unique identifiers for the root nodes. Also returns htree, which is an HTuple object with mess and emask attributes initialized used for masking/identifying root related info while node and vmask are None.

        hinter - HTuple object; Here fmess is from tree_tensors/inter_tensors.
            mess - tuple (h,c) initialized with zeros of size (len(fmess), hidden_size) for both.
            emask - tensor of zeros of size (len(fmess))
        
        hgraph - HTuple object; Here fmess is from graph_tensors.
            mess - tuple (h,c) initialized with zeros of size (len(fmess), hidden_size) for both.
            vmask - tensor of zeros of size = len(fnode)
            emask - tensor of zeros of size = len(fmess)

        all_topo_preds, all_cls_preds, all_assm_preds, new_atoms - these are all initialized as empty lists to store the predictions of the model.
        tree_scope - stores the tuples of index of root nodes in the tree_batch and the corresponding metal complex length. 

        """

        batch_size = len(orders)
        tree_batch, graph_batch = graphs
        tree_tensors, graph_tensors = tensors

        inter_tensors = tree_tensors
        # distance_true_tree = tree_tensors[1][:, 4]
        # atom_distances=graph_tensors[1][:, 4]
        tree_hmess=tree_tensors[1]
        atom_hmess=graph_tensors[1]

        src_root_vecs, src_tree_vecs, src_graph_vecs = src_mol_vecs
        init_vecs = src_root_vecs if self.latent_size == self.hidden_size else self.W_root(src_root_vecs) # just a linear layer

        htree, htree_motif, tree_tensors, tree_tensors_motif = self.init_decoder_state(tree_batch, tree_tensors, init_vecs)
        inter_tensors_motif=inter_tensors
        graph_tensors_motif=graph_tensors
        hinter = HTuple(
            mess = self.rnn_cell.get_init_state(inter_tensors[1]),
            emask = self.itensor.new_zeros(inter_tensors[1].size(0))
        )
        hinter_motif = HTuple(
            mess = self.rnn_cell_motif.get_init_state(inter_tensors_motif[1]),
            emask = self.itensor.new_zeros(inter_tensors_motif[1].size(0))
        )
        hgraph = HTuple(
            mess = self.rnn_cell.get_init_state(graph_tensors[1]),
            vmask = self.itensor.new_zeros(graph_tensors[0].size(0)),
            emask = self.itensor.new_zeros(graph_tensors[1].size(0))
        )
        hgraph_motif = HTuple(
            mess = self.rnn_cell_motif.get_init_state(graph_tensors_motif[1]),
            vmask = self.itensor.new_zeros(graph_tensors_motif[0].size(0)),
            emask = self.itensor.new_zeros(graph_tensors_motif[1].size(0))
        )
        
        all_topo_preds, all_cls_preds, all_assm_preds = [], [], []
        new_atoms = []
        tree_scope = tree_tensors[-1]

        """
        root - it is assigned the root node from the tree_batch corresponding to the index stored in the tree_scope.

        clab - cluster index from hmap of pairVocab corresponding to the root node's smiles(motif). 
        ilab - tuple index from vmap of pairVocab corresponding to the root node's (smiles,ismiles).

        all_cls_preds - list of tuples (root_vec_encoded, batch_idx, clab, ilab) for each root node in the batch.

        new_atoms - list of atoms (flattened) belonging to this cluster(motif) with offset to account for the tree_batch corresponding to the root nodes of the batch.

        For our case, since the root node is the iron atom for every complex, clab and ilab will always be the same for all the root nodes. New atoms will only consist of the iron atom itself set by the graph(Atom) level offset. 

        The code is performing cluster predictions by mapping the labels of root nodes (from tree_scope) to cluster identifiers (clab and ilab) using a vocabulary (self.vocab). While it processes the same nodes from tree_scope, the purpose is to predict their cluster labels and prepare these predictions for further processing.
        """
        root_motifs=[]
        for i in range(batch_size):
            root = tree_batch.nodes[ tree_scope[i][0] ]
            clab, ilab = self.vocab[ root['label'] ] # label here is (smiles,ismiles) for the partiicular node. vocab maps it to a unique index for the cluster (clab) and another unique index for the tuple (ilab)
            all_cls_preds.append( (init_vecs[i], i, clab, ilab) ) #cluster prediction
            root_motifs.append(tree_scope[i][0])
            new_atoms.extend(root['cluster'])
        
        root_atoms = new_atoms.copy()
        """
        subgraph

            INPUT: graph_batch (from tensorize), new_atoms (atoms belonging to the clusters of the root nodes - Iron everytime for our case), hgraph (The HTuple object initialized with mess, vmask, emask attributes)
            OUTPUT: tuple of (new_atom_index, new_bond_index) - tensors of the same dtype and device as vmask and emask respectively containing the indices of the new_atoms and the corresponding new_bonds between any of the atoms in the new_atoms list already existing in graph_batch. 
        
        graph_tensors

            uses embed_graph from the encoder to process the graph_tensors. It is then concatenated with the last element of the original graph_tensors which is just the scope of the graph_tensors.

        maxt - maximum length of the orders list in the batch.
        max_cls_size - maximum size of the clusters in the batch. It is calculated by taking the maximum of the length of the cluster of each node in the tree_batch and multiplying it by 2.

        graph_tensors_atom and hgraph_atom has been added newly to handle the distance prediction at the atom level. It is processed in the same way as graph_tensors.

        """
        
        subgraph = self.update_graph_mask(graph_batch, new_atoms, hgraph)
        subgraph_motif = self.update_graph_mask(graph_batch, new_atoms, hgraph_motif)
        graph_tensors_atom=graph_tensors
        graph_tensors = self.hmpn.embed_graph(graph_tensors) + (graph_tensors[-1],) #preprocess graph tensors
        graph_tensors_motif = self.hmpn_motifdist.embed_graph(graph_tensors_motif) + (graph_tensors_motif[-1],)
        hgraph_atom=HTuple(
            mess = self.rnn_cell_atom.get_init_state(graph_tensors_atom[1]),
            vmask = self.itensor.new_zeros(graph_tensors_atom[0].size(0)),
            emask = self.itensor.new_zeros(graph_tensors_atom[1].size(0))
        )
        graph_tensors_atom = self.hmpn_distances.embed_graph(graph_tensors_atom) + (graph_tensors_atom[-1],) #preprocess graph tensors

        #*New message passing network needed for edge distance prediction. Since using the same message passing network for both cluster prediction and the distance prediction between motifs is not working well since it is not able to focus on one task and is trying to balance itout between both the tasks thereby not perfecting either of them.



        maxt = max([len(x) for x in orders])
        max_cls_size = max( [len(attr) * 2 for node,attr in tree_batch.nodes(data='cluster')] )

        """
        The loop iterates for the maximum number of steps (maxt) to handle all traversal sequences in orders.

        batch_list - contains the indices of trees in the batch that still have steps to process at time t. Ensures only trees that need updates at this step are processed.

        subtree: For each tree in batch_list, retrieves the current node (xid), its child node (yid), and the label (tlab) from the dfs order. Adds xid to subtree[0] and message indices (mess_idx) to subtree[1] if a child node exists (yid is not None).

        Converts subtree lists into tensors for further processing.
        
        emask.scater - Updates htree and hinter masks to indicate which edges are active in this step.

        Applies the updated masks to extract tensors (cur_tree_tensors, cur_inter_tensors, and cur_graph_tensors) that are relevant to the current step.

        Uses MPN to update the states htree, hinter, hgraph. These can be used to retrieve the hidden state vectors which can be passed through a linear layer to predict the distance information at the tree level. It dosent seem like using htree or hinter would make a difference for predicting the distance for that edge but we will try both and see which one is converging. 
        """

        # New prediction of distances
        distance_preds_tree = []
        distance_true_tree = torch.empty(0)

        flagged_motifs = set()  # To store flagged edges for future checks
        flagged_edges = set()
        distance_preds_graph=[]
        distance_true_graph=torch.empty(0)

        edge_distances={}
        for edge in atom_hmess:
            key = (int(edge[0].item()), int(edge[1].item()))  # Keep direction (a, b)
            edge_distances[key] = edge[4]
        
        edge_distances_motifs={}
        for edge in tree_hmess:
            key = (int(edge[0].item()), int(edge[1].item()))
            edge_distances_motifs[key] = edge[4]

        output = []
        output_clusters = []
        output_distance = []

        for t in range(maxt):
            batch_list = [i for i in range(batch_size) if t < len(orders[i])]
            assert htree.emask[0].item() == 0 and hinter.emask[0].item() == 0 and hgraph.vmask[0].item() == 0 and hgraph.emask[0].item() == 0

            subtree = [], []
            for i in batch_list:
                xid, yid, tlab = orders[i][t]
                subtree[0].append(xid) #storing the parent 
                if yid is not None: # if child is present. 
                    mess_idx = tree_batch[xid][yid]['mess_idx']
                    subtree[1].append(mess_idx)

            # #print("subtree: ", subtree)
            # exit()

            subtree_motif = subtree
            subtree = htree.emask.new_tensor(subtree[0]), htree.emask.new_tensor(subtree[1]) 
            subtree_motif = htree_motif.emask.new_tensor(subtree_motif[0]), htree_motif.emask.new_tensor(subtree_motif[1])

            htree.emask.scatter_(0, subtree[1], 1)
            hinter.emask.scatter_(0, subtree[1], 1)

            htree_motif.emask.scatter_(0,subtree_motif[1],1)
            hinter_motif.emask.scatter_(0,subtree_motif[1],1)

            cur_tree_tensors = self.apply_tree_mask(tree_tensors, htree, hgraph)
            cur_inter_tensors = self.apply_tree_mask(inter_tensors, hinter, hgraph)
            cur_graph_tensors = self.apply_graph_mask(graph_tensors, hgraph)

            motif_tree_tensors = self.apply_tree_mask(tree_tensors_motif, htree_motif, hgraph_motif)
            motif_inter_tensors = self.apply_tree_mask(inter_tensors_motif, hinter_motif, hgraph_motif)
            motif_graph_tensors = self.apply_graph_mask(graph_tensors_motif, hgraph_motif)
            # print("current tree tensors mess")
            # print(cur_tree_tensors[1])
            # print("current graph tensors mess")
            # print(cur_graph_tensors[1])
            htree, hinter, hgraph = self.hmpn(cur_tree_tensors, cur_inter_tensors, cur_graph_tensors, htree, hinter, hgraph, subtree, subgraph) 

            htree_motif, hinter_motif, hgraph_motif = self.hmpn_motifdist(motif_tree_tensors, motif_inter_tensors, motif_graph_tensors, htree_motif, hinter_motif, hgraph_motif, subtree_motif, subgraph_motif)
            # print(htree.mess)
            # print(hgraph)
            # exit()

            """
            -> Topo Preds - Records predictions for the current tree topology (all_topo_preds).

            -> A new list of new_atoms is constructed which is the child nodes of the current time step's node from the order. It is to remake the subgraph for the next time step t.

            if tlab==0 - means the dfs order has come to an end. Now it will just backtrack so the order has reached an end and the loop will continue to the next tree in the batch. 

            -> Cluster Preds - Records predictions for the current tree cluster (all_cls_preds).

            -> distance prediction - uses a linear layer to predict the distance between the nodes corresponding to that edge using the hidden state of the message.


            """
            # ! normalization can maybe help to converge. If we can normalize the coordinates in a range of 0 to 1. 
            new_atoms = []
            prediction_atoms=[]
            # #print("htree.node:",htree.node[1])
            # exit()
            for i in batch_list:
                xid, yid, tlab = orders[i][t]

                all_topo_preds.append( (htree.node[xid], i, tlab) ) #topology prediction
                e_dist=None
                if yid is not None:
                    mess_idx = tree_batch[xid][yid]['mess_idx']
                    new_atoms.extend( tree_batch.nodes[yid]['cluster'] ) #NOTE: regardless of tlab = 0 or 1
                    # distance prediction at the motif level first and then at the atom level. 
                    hmess_motif = self.rnn_cell_motif.get_hidden_state(htree_motif.mess)
                    e_dist=self.W_dist_motifs(hmess_motif[mess_idx])
                    distance_preds_tree.append(e_dist)
                    distance_true_tree = torch.cat([distance_true_tree, edge_distances_motifs[(xid,yid)].unsqueeze(0)])
                    #print("prediction for tree level motif", e_dist)
                    # exit()

                    # retrieving the graph edges from motifs. 
                    yid_atoms=tree_batch.nodes[yid]['cluster']
                    xid_atom=tree_batch.nodes[xid]['cluster']

                    # edges between root xid and yid_atoms
                    root_edges=[]
                    root_distances = []
                    if xid_atom[0] in root_atoms:

                        for atom in yid_atoms:
                            fkey = (xid_atom[0], atom)  # Forward key
                            bkey = (atom, xid_atom[0])  # Backward key
                            
                            if fkey in edge_distances:
                                # root_edges.append(graph_batch[xid_atom[0]][atom]['mess_idx'])
                                root_edges.append(fkey)
                                root_distances.append(edge_distances[fkey])
                                prediction_atoms.append(xid_atom[0])
                                prediction_atoms.append(atom)

                    #within yid_atoms
                    internal_edges=[]
                    internal_distances=[]
                    if yid not in flagged_motifs:
                        internal_edges = [
                            tuple(sorted((int(src.item()), int(tgt.item())))) for src, tgt, *_ in atom_hmess
                            if src in yid_atoms and tgt in yid_atoms and yid not in flagged_motifs
                        ]
                        internal_edges = list(set(internal_edges))
                        for edge in internal_edges:
                            prediction_atoms.append(edge[0])
                            prediction_atoms.append(edge[1])
                            internal_distances.append(edge_distances[edge])

                    # Flag them to avoid reprocessing while backtracking in dfs. Only necessary for edges within yid_atoms
                    flagged_motifs.add(yid)

                    # now we update the hgraph_atom for the prediction atoms.
                    subgraph_atoms=self.update_graph_mask(graph_batch,prediction_atoms,hgraph_atom)
                    prediction_graph_tensors=self.apply_graph_mask(graph_tensors_atom,hgraph_atom)
                    hgraph_atom=self.hmpn_distances(prediction_graph_tensors, hgraph_atom, subgraph_atoms)
                    
                    for j,key in enumerate(root_edges):
                        # distance prediction. 
                        mess_idx=graph_batch[key[0]][key[1]]['mess_idx']
                        hmess=self.rnn_cell_atom.get_hidden_state(hgraph_atom.mess)

                        e_dist_atom=self.W_dist(hmess[mess_idx])
                        distance_preds_graph.append(e_dist_atom)
                        # distance_preds_graph.append(e_dist_atom)
                        # distance_true_graph.append(root_distances[j])
                        distance_true_graph = torch.cat([distance_true_graph, edge_distances[key].unsqueeze(0)])

                        #print(f"prediction is {e_dist_atom} and true is {edge_distances[key].unsqueeze(0)}")

                    for j,edge in enumerate(internal_edges):
                        mess_idx=graph_batch[edge[0]][edge[1]]['mess_idx']
                        hmess=self.rnn_cell_atom.get_hidden_state(hgraph_atom.mess)
                        e_dist_atom=self.W_dist(hmess[mess_idx])
                        # distance_preds_graph.append(e_dist_atom)
                        distance_preds_graph.append(e_dist_atom)
                        # distance_true_graph.append(internal_distances[j])
                        distance_true_graph = torch.cat([distance_true_graph, edge_distances[edge].unsqueeze(0)])

                        #print(f"prediction is {e_dist_atom} and true is {edge_distances[edge].unsqueeze(0)}")
                                    
                if tlab == 0: 
                    continue
                    
                    

                cls = tree_batch.nodes[yid]['smiles']
                clab, ilab = self.vocab[ tree_batch.nodes[yid]['label'] ]
                mess_idx = tree_batch[xid][yid]['mess_idx']
                hmess = self.rnn_cell.get_hidden_state(htree.mess)
                all_cls_preds.append( (hmess[mess_idx], i, clab, ilab) ) #cluster prediction using message                
                output.append((len(all_topo_preds)-1,len(all_cls_preds)-1, orders[i][t]))
                output_distance.append((len(distance_preds_tree)-1, i))

                inter_label = tree_batch.nodes[yid]['inter_label']
                inter_label = [ (pos, self.vocab[(cls, icls)][1]) for pos,icls in inter_label ]
                inter_size = self.vocab.get_inter_size(ilab)

                if len(tree_batch.nodes[xid]['cluster']) > 2: #uncertainty occurs only when previous cluster is a ring
                    nth_child = tree_batch[yid][xid]['label'] #must be yid -> xid (graph order labeling is different from tree)
                    cands = tree_batch.nodes[yid]['assm_cands']
                    icls = list(zip(*inter_label))[1]
                    cand_vecs = self.enum_attach(hgraph, cands, icls, nth_child)

                    if len(cand_vecs) < max_cls_size:
                        pad_len = max_cls_size - len(cand_vecs)
                        cand_vecs = F.pad(cand_vecs, (0,0,0,pad_len))

                    batch_idx = hgraph.emask.new_tensor( [i] * max_cls_size )
                    all_assm_preds.append( (cand_vecs, batch_idx, 0) ) #the label is always the first of assm_cands

            subgraph = self.update_graph_mask(graph_batch, new_atoms, hgraph)
            subgraph_motif = self.update_graph_mask(graph_batch, new_atoms, hgraph_motif)

        """
        I think here src_tree_vecs is being used for the loss computation. and we must pass it from the encoded part for loss computation. Instead, right now we are just passing the latent vectors for all three levels. 
        """
        topo_vecs, batch_idx, topo_labels = zip_tensors_metal(all_topo_preds)
        topo_scores = self.get_topo_score(src_tree_vecs, batch_idx, topo_vecs)
        topo_loss = self.topo_loss(topo_scores, topo_labels.float())
        topo_acc = get_accuracy_bin(topo_scores, topo_labels)

        cls_vecs, batch_idx, cls_labs, icls_labs = zip_tensors_metal(all_cls_preds)
        cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, batch_idx, cls_vecs, cls_labs)

        cls_loss = self.cls_loss(cls_scores, cls_labs) + self.icls_loss(icls_scores, icls_labs)
        cls_acc = get_accuracy(cls_scores, cls_labs)
        icls_acc = get_accuracy(icls_scores, icls_labs)

        predicted_tensor = torch.cat(distance_preds_graph, dim=0)
        assert predicted_tensor.shape == distance_true_graph.shape, f"Shape mismatch: {predicted_tensor.shape} vs {distance_true_graph.shape}"
        distloss = self.dist_loss(predicted_tensor, distance_true_graph)

        predicted_tensor_tree = torch.cat(distance_preds_tree, dim=0)
        assert predicted_tensor_tree.shape == distance_true_tree.shape, f"Shape mismatch: {predicted_tensor_tree.shape} vs {distance_true_tree.shape}"
        distloss_tree = self.dist_loss_motifs(predicted_tensor_tree, distance_true_tree)
        # dist_loss_atoms=self.dist_loss(torch.stack(distance_preds_graph).squeeze(),torch.stack(distance_true_graph).squeeze())

        # dist_loss=self.dist_loss(torch.cat(distance_preds_tree,dim=0),distance_true_tree[1:])
        # exit()

        # print(distance_true_tree[1:])
        # exit()

        ligand_count_pred = []
        for i, root_vec in enumerate(src_root_vecs):
            ligand_logits = self.ligand_classifier(root_vec)
            ligand_count_pred.append(ligand_logits.unsqueeze(0))

        predicted_ligandcount = torch.cat(ligand_count_pred)
        true_ligandcount = torch.tensor(ligand_counts, dtype=torch.float32) 
        true_ligandcount = true_ligandcount.long()

        count_loss = self.classification_loss(predicted_ligandcount, true_ligandcount)      

        if len(all_assm_preds) > 0:
            assm_vecs, batch_idx, assm_labels = zip_tensors_metal(all_assm_preds)
            assm_scores = self.get_assm_score(src_graph_vecs, batch_idx, assm_vecs)
            assm_loss = self.assm_loss(assm_scores, assm_labels)
            assm_acc = get_accuracy_sym(assm_scores, assm_labels)
        else:
            assm_loss, assm_acc = 0, 1

        # if(epoch_no==50):
        #     print("-------------------------------------------------")
        #     for i in range(len(cls_labs)):
        #         print(f"Predicted Smiles: {self.vocab.get_smiles(cls_scores[i].max(dim=-1)[1].item())} and True Smiles: {self.vocab.get_smiles(cls_labs[i])}")
        #         print(f"Predicted Ismiles: {self.vocab.get_ismiles(icls_scores[i].max(dim=-1)[1].item())} and True Ismiles: {self.vocab.get_ismiles(icls_labs[i])}")
        #         print("-------------------------------------------------")
                
        #     import networkx as nx
        #     import matplotlib.pyplot as plt 

        #     graphs = {0: nx.DiGraph(), 1: nx.DiGraph(), 2: nx.DiGraph()}

        #     # Process each distance entry
        #     for i,node in enumerate(root_motifs):
        #         print(node)
        #         graphs[i].add_node(node, label=("Fe", "Fe:2"))

        #     global_index=0
        #     for idx, batch in output_distance:
        #         pred_distance = distance_preds_tree[idx]
        #         true_distance = distance_true_tree[idx]

        #         # Get DFS order
        #         parent, child, _ = output[global_index][2]
        #         print(f"Parent: {parent} and Child: {child}")

        #         # Get node labels
        #         child_smiles = self.vocab.get_smiles(cls_scores[output[global_index][1]].max(dim=-1)[1].item())
        #         child_ismiles = self.vocab.get_ismiles(icls_scores[output[global_index][1]].max(dim=-1)[1].item())
            
        #         # Add nodes to the corresponding batch graph
        #         graphs[batch].add_node(child, label=f"Predicted : {(child_smiles, child_ismiles)}, True: {(self.vocab.get_smiles(cls_labs[output[global_index][1]].item()), self.vocab.get_ismiles(icls_labs[output[global_index][1]].item()))}")

        #         # Add edge with predicted distance as label
        #         graphs[batch].add_edge(parent, child, label=f"Predicted Dist: {pred_distance.item():.2f}, True Dist: {true_distance.item():.2f}")

        #         # graphs[batch].add_edge(parent, child, label=f"Predicted Dist: {pred_distance:.2f}, True Dist: {true_distance:.2f}")
        #         global_index += 1

        #     import networkx as nx
        #     import plotly.graph_objects as go
        #     import numpy as np

        #     def plot_3d_graph(graph, title):
        #         # Get nodes and edges
        #         pos = nx.spring_layout(graph, dim=3)  # 3D layout
        #         node_x, node_y, node_z = [], [], []
                
        #         # Collect node positions
        #         for node, (x, y, z) in pos.items():
        #             node_x.append(x)
        #             node_y.append(y)
        #             node_z.append(z)

        #         # Collect edge positions
        #         edge_x, edge_y, edge_z = [], [], []
        #         edge_labels = []
        #         for edge in graph.edges(data=True):
        #             x0, y0, z0 = pos[edge[0]]
        #             x1, y1, z1 = pos[edge[1]]
        #             edge_x.extend([x0, x1, None])
        #             edge_y.extend([y0, y1, None])
        #             edge_z.extend([z0, z1, None])
        #             edge_labels.append(f"{edge[2]['label']}")  # Edge label as distance

        #         # Create trace for edges
        #         edge_trace = go.Scatter3d(
        #             x=edge_x, y=edge_y, z=edge_z,
        #             line=dict(width=2, color='black'),
        #             hoverinfo='none',
        #             mode='lines'
        #         )

        #         # Create trace for nodes
        #         node_trace = go.Scatter3d(
        #             x=node_x, y=node_y, z=node_z,
        #             mode='markers+text',
        #             text=[graph.nodes[n]["label"] for n in graph.nodes],
        #             marker=dict(size=8, color="blue", opacity=0.8),
        #         )

        #         # Create layout and plot
        #         fig = go.Figure(data=[edge_trace, node_trace])
        #         fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40))
        #         fig.show()


        #     for batch in range(3):
        #         plot_3d_graph(graphs[batch], f"Graph for Batch {batch}")

            # print("Epoch 43")
            # print("-------------------------------------------------")
            # print("graph level distances")
            # for i in range(len(distance_preds_graph)):
            #     print(f"Predicted: {distance_preds_graph[i]} and True: {distance_true_graph[i]}")
            # print("-------------------------------------------------")
            # print("motif level distances")
            # for i in range(len(distance_preds_tree)):
            #     print(f"Predicted: {distance_preds_tree[i]} and True: {distance_true_tree[i]}")
            # for i, curr in enumerate(output):
            #     print(f"Topo score: {topo_scores[curr[0]]} and Topo label: {topo_labels[curr[0]]}")
            #     print(f"Cls label: {self.vocab.get_smiles(cls_scores[curr[1]].max(dim=-1)[1].item())}")
            #     print(f"Icls label: {self.vocab.get_ismiles(icls_scores[curr[1]].max(dim=-1)[1].item())}")  
            #     print(f"Original order: {curr[2]}")
            #     print("-------------------------------------------------")
        loss = (topo_loss + cls_loss + assm_loss + distloss + distloss_tree + count_loss) / batch_size
        return loss, cls_acc, icls_acc, topo_acc, assm_acc, distloss, distloss_tree, count_loss

    def enum_attach(self, hgraph, cands, icls, nth_child):
        cands = self.itensor.new_tensor(cands)
        icls_vecs = self.itensor.new_tensor(icls * len(cands))
        icls_vecs = self.E_assm( icls_vecs )

        nth_child = self.itensor.new_tensor([nth_child] * len(cands.view(-1)))
        order_vecs = self.E_order.index_select(0, nth_child)

        cand_vecs = hgraph.node.index_select(0, cands.view(-1))
        cand_vecs = torch.cat( [cand_vecs, icls_vecs, order_vecs], dim=-1 )
        cand_vecs = self.matchNN(cand_vecs)

        if len(icls) == 2:
            cand_vecs = cand_vecs.view(-1, 2, self.hidden_size).sum(dim=1)
        return cand_vecs

    def decode(self, src_mol_vecs, greedy=True, max_decode_step=100, beam=5):
        """
        src_mol_vecs - random vectors of latent_size of size equal to batch_size
        batch_size - number of molecules to generate
        tree_batch - IncTreeMetal object - tree representation of the molecule, it is an empty graph initially
        graph_batch - IncGraphMetal object - graph representation of the molecule, it is an empty graph initially. Here the node_fdim is passed as the size of avocab and the edge_fdim is passed as the sum of the size of avocab and bond_size (len of bond_list + 1 + len of fdim_object) of the hmpn object.

        The initial root clusters are retrieved using get_cls_score and choosing the max probability of the cluster. This is done for each molecule in the batch_size.
        """
        src_root_vecs, src_tree_vecs, src_graph_vecs = src_mol_vecs
        batch_size = len(src_root_vecs)

        tree_batch = IncTreeMetal(batch_size, node_fdim=2, edge_fdim=4) # the extra dimension is for the distance
        graph_batch = IncGraphMetal(self.avocab, batch_size, node_fdim=self.hmpn.atom_size, edge_fdim=self.hmpn.atom_size + self.hmpn.bond_size)
        stack = [[] for i in range(batch_size)]

        init_vecs = src_root_vecs if self.latent_size == self.hidden_size else self.W_root(src_root_vecs)
        batch_idx = self.itensor.new_tensor(range(batch_size))
        cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, batch_idx, init_vecs, None)
        root_cls = cls_scores.max(dim=-1)[1] # extracts the index of the max probability cluster for each sample in the batch. 
        icls_scores = icls_scores + self.vocab.get_mask(root_cls)
        root_cls, root_icls = root_cls.tolist(), icls_scores.max(dim=-1)[1].tolist()

        """
        This super root is the root of the tree_batch. It is the parent of all the root nodes of the molecules in the batch.
        
        The loop below creates the root nodes of the molecules in the batch and adds them to the tree_batch. Also adds an edge to the super_root. The motif added to the tree_batch is taken and its smiles is retrieved from the vocabulary that we previously created. The mol object from that smiles is created and then the individual atoms and bonds are added to the graph_batch. 

        ! we need to change this because we are not really predicting any cluster for the root node. We are fixing it to be the iron atom/motif for all the molecules in the batch. When we expand the dataset to multiple metals, we can have a technique where we predict the root node among those metals. Maybe something like maintaing a different vocab for just the root nodes or if the network is strong enough, it can predict the root node to be the metals only from the latent space. Whenever it is not a metal, we will discard it. 

        ^ to make it a utility for chemists, create a function which will let us choose the central metal atom from a dictionay of different type of metals. Also do it without any condition on the larger dataset and see if the network can predict the central metal atom.

        """
        super_root = tree_batch.add_node()
        print("batch size : ", batch_size)
        root_indices_tree=[]
        root_indices_graph=[]
        for bid in range(batch_size):
            clab, ilab = self.vocab.size()[0]-1, self.vocab.size()[1]-1
            root_idx = tree_batch.add_node( batch_idx.new_tensor([clab, ilab]) )
            root_indices_tree.append(root_idx)
            tree_batch.add_edge(super_root, root_idx)
            stack[bid].append(root_idx)

            root_smiles = self.vocab.get_ismiles(ilab)
            print("Root smiles :", root_smiles)
            new_atoms, new_bonds, attached, new_highlight_atoms = graph_batch.add_mol(bid, root_smiles, [], 0)
            root_indices_graph.extend(new_atoms)
            print("Highlight_atoms : ", new_highlight_atoms)
            tree_batch.register_cgraph(root_idx, new_atoms, new_bonds, attached, new_highlight_atoms)
            print(attached)    

        print("Root indices of the graph",root_indices_graph)
        print("Root indices of the tree",root_indices_tree)
        # exit(0)
        # for bid in range(batch_size):
        #     clab, ilab = root_cls[bid], root_icls[bid]
        #     root_idx = tree_batch.add_node( batch_idx.new_tensor([clab, ilab]) )
        #     tree_batch.add_edge(super_root, root_idx) 
        #     stack[bid].append(root_idx)

        #     root_smiles = self.vocab.get_ismiles(ilab)
        #     new_atoms, new_bonds, attached = graph_batch.add_mol(bid, root_smiles, [], 0)
        #     tree_batch.register_cgraph(root_idx, new_atoms, new_bonds, attached)
        
        #invariance: tree_tensors is equal to inter_tensors (but inter_tensor's init_vec is 0)
        # ! currently the tree_tensors consists of the super root node and the root nodes (i.e. all iron nodes) of the molecules in the batch. The same goes for graph_tensors.
        tree_tensors = tree_batch.get_tensors()
        graph_tensors = graph_batch.get_tensors()

        htree = HTuple( mess = self.rnn_cell.get_init_state(tree_tensors[1]) )
        hinter = HTuple( mess = self.rnn_cell.get_init_state(tree_tensors[1]) )
        hgraph = HTuple( mess = self.rnn_cell.get_init_state(graph_tensors[1]) )

        # ^ new tensors for the motif distance prediction.
        tree_tensors_motif = tree_batch.get_tensors()
        graph_tensors_motif = graph_batch.get_tensors()

        htree_motif = HTuple( mess = self.rnn_cell_motif.get_init_state(tree_tensors[1]) )
        hinter_motif = HTuple( mess = self.rnn_cell_motif.get_init_state(tree_tensors[1]) )
        hgraph_motif = HTuple( mess = self.rnn_cell_motif.get_init_state(graph_tensors[1]) )
        # ^
        # ^ new tensors for the atom distance prediction.
        graph_tensors_atom=graph_batch.get_tensors()
        hgraph_atom=HTuple(mess = self.rnn_cell_atom.get_init_state(graph_tensors_atom[1]))

        h = self.rnn_cell.get_hidden_state(htree.mess)
        h[1 : batch_size + 1] = init_vecs #wiring root (only for tree, not inter)

        #^ Predicting the number of ligands for each complex using the latent space vectors.
        # ! Not necessary for now since dfs ensures the entire ligand is traversed before returning to the root. 
        ligand_count_pred = {}
        for i, root_vec in enumerate(src_root_vecs):
            n_ligands = self.ligand_classifier(root_vec)
            ligand_count_pred[i] = n_ligands.argmax().item()
        
        print("ligand count prediction: ", ligand_count_pred)
        print("Initial stack", stack)
        #^
        
        # 100 molecules to predict
        """
        1 - 3
        2 - 4
        3 - 5

        for i in range(upper_bound):
            decrement the dictionary
            
        
        """
        for t in range(max_decode_step):
            """
            if there are any nodes in the stack, only those samples are considered for further processing.

            the top node of the stack is taken and the subtree is created with the current node and its child nodes (empty initially). The subtree is then converted to tensors for further processing.
            subgraph consists of the nodes and edges at atom level for the current tree nodes.

            message passing is done on the tree and graph tensors of the root nodes (in our case it will be only iron for now.) using the hmpn function. The updated states are stored in htree, hinter, hgraph.

            after message passing, the topological score predictions are computed for the current tree nodes and the latent space vectors corresponding to that particular batch id. these scores are converted to probabilities using sigmod function. 
            """
            batch_list = [ bid for bid in range(batch_size) if len(stack[bid]) > 0]
            if len(batch_list) == 0: break

            # ^ Checking if one ligand predictions has been completed, if it has, then we connect all the unconnected :2 atoms to the iron node of that complex. 
            for bid in batch_list:
                # ^ the second condition is put because otherwise it will again try to connect when there are no new ligands but the iron atom remains in the stack. 
                if stack[bid][-1] in root_indices_tree and ligand_count_pred[bid]>0: 
                    graph_batch.connect_ligand(bid, stack[bid][-1], root_indices_graph[bid])
                    # tree_batch.update_edges(stack[bid][-1], new_bonds)
                    ligand_count_pred[bid] -= 1
            
            # ^ Mofifying the batch list based on whether more ligands need to be predicted for those complexes or not.
            batch_list = [ bid for bid in batch_list if ligand_count_pred[bid] > 0]
            if len(batch_list) == 0: break


            batch_idx = batch_idx.new_tensor(batch_list)
            cur_tree_nodes = [stack[bid][-1] for bid in batch_list]
            subtree = batch_idx.new_tensor(cur_tree_nodes), batch_idx.new_tensor([])
            subgraph = batch_idx.new_tensor( tree_batch.get_cluster_nodes(cur_tree_nodes) ), batch_idx.new_tensor( tree_batch.get_cluster_edges(cur_tree_nodes) )

            subtree_motif = batch_idx.new_tensor(cur_tree_nodes), batch_idx.new_tensor([])
            subgraph_motif = batch_idx.new_tensor( tree_batch.get_cluster_nodes(cur_tree_nodes) ), batch_idx.new_tensor( tree_batch.get_cluster_edges(cur_tree_nodes) )

            htree, hinter, hgraph = self.hmpn(tree_tensors, tree_tensors, graph_tensors, htree, hinter, hgraph, subtree, subgraph)
            htree_motif, hinter_motif, hgraph_motif = self.hmpn_motifdist(tree_tensors_motif, tree_tensors_motif, graph_tensors_motif, htree_motif, hinter_motif, hgraph_motif, subtree_motif, subgraph_motif)

            print("first message passing is done")

            topo_scores = self.get_topo_score(src_tree_vecs, batch_idx, htree.node.index_select(0, subtree[0]))
            topo_scores = torch.sigmoid(topo_scores)
            if greedy:
                topo_preds = topo_scores.tolist()
            else:
                topo_preds = torch.bernoulli(topo_scores).tolist()

            new_mess = []
            new_motif_bond = {}
            new_motif_atom_pair = {}
            expand_list = []

            """
            for each instance of the batch_size, it checks if the topology prediction is greater than 0.5 and if the tree can be expanded (checks if the indegree is less than the max number of neighbors). If yes, it expands the tree by adding a new node (the feature/label is not added yet, just the node.) and an edge from the parent to the child. If not, it pops the node and adds an edge from the child to the parent (the backtrack edge with the edge feature containing the position of the child similar to the way that was done during the preprocessing).
            """
            for i,bid in enumerate(batch_list):
                if topo_preds[i] > 0.5 and tree_batch.can_expand(stack[bid][-1]):
                    expand_list.append( (len(new_mess), bid) )
                    new_node = tree_batch.add_node() #new node label is yet to be predicted
                    edge_feature = batch_idx.new_tensor( [stack[bid][-1], new_node, 0, 0] ) #parent to child is 0 #^ new index in this feature at the last is for future distance prediction. 
                    new_edge = tree_batch.add_edge(stack[bid][-1], new_node, edge_feature) 
                    stack[bid].append(new_node)
                    new_mess.append(new_edge)
                    new_motif_atom_pair[bid] = (stack[bid][-2], stack[bid][-1]) #parent and child
                    new_motif_bond[bid] = new_edge
                else:
                    child = stack[bid].pop()
                    if len(stack[bid]) > 0:
                        nth_child = tree_batch.graph.in_degree(stack[bid][-1]) #edge child -> father has not established
                        #^ the edge feature's distance column while backtracking is set to the same one as predicted previously during forward propagation. We dont want to predict a different distance for the same pair of nodes in the tree.
                        edge_distance = tree_batch.fmess[tree_batch.edge_dict[(stack[bid][-1], child)]][-1]
                        edge_feature = batch_idx.new_tensor( [child, stack[bid][-1], nth_child, edge_distance] )
                        new_edge = tree_batch.add_edge(child, stack[bid][-1], edge_feature)
                        new_mess.append(new_edge)
            """
            Now, again message passing is done. In the second message passing, only the subtree (tree nodes) is updated, and subgraph = ([], []), meaning no new graph-level information is added. The reason for this is likely that the graph connections (external to the tree) remain unchanged at this point. Since the new edges are only being added within the tree, message passing focuses on the new local updates in the tree rather than re-processing the entire graph.
            """
            subtree = subtree[0], batch_idx.new_tensor(new_mess) 
            subgraph = [], []
            htree, hinter, hgraph = self.hmpn(tree_tensors, tree_tensors, graph_tensors, htree, hinter, hgraph, subtree, subgraph)

            # subtree_motif = subtree_motif[0], batch_idx.new_tensor(new_mess)
            # subgraph_motif = [], []

            cur_mess = self.rnn_cell.get_hidden_state(htree.mess).index_select(0, subtree[1])
            # cur_mess_motifs = self.rnn_cell_motif.get_hidden_state(htree_motif.mess).index_select(0, subtree_motif[1])
            print("second message passing is done")
            # exit(0)

            """
            The code below involves only the new edges discovered in the forward pass and not the backward edges i.e. only the expand_list. For the forward new edges that are discovered in the previous step, those hidden state mess tensors are taken and they are now used along with the latent space vectors to calculate cls and icls scores for each of the corresponding new edge. Now these cls_scores and icls_scores are sent to the hier_topk function and the combined scores along with top-k individual scores of cls and icls are returned. 
            """
            if len(expand_list) > 0:
                idx_in_mess, expand_list = zip(*expand_list)
                idx_in_mess = batch_idx.new_tensor( idx_in_mess )
                expand_idx = batch_idx.new_tensor( expand_list )
                forward_mess = cur_mess.index_select(0, idx_in_mess)
                cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, expand_idx, forward_mess, None)
                scores, cls_topk, icls_topk = hier_topk_metal(cls_scores, icls_scores, self.vocab, beam)
                if not greedy:
                    scores = torch.exp(scores) #score is output of log_softmax
                    shuf_idx = torch.multinomial(scores, beam, replacement=True).tolist()
            
            new_bonds_expandlist = defaultdict(list)
            for i,bid in enumerate(expand_list):
                new_node, fa_node = stack[bid][-1], stack[bid][-2]

                # ^ Check to know if we are predicting primary motif or not. 
                if fa_node in root_indices_tree:
                    primary_motif = True
                else:
                    primary_motif = False

                success = False
                cls_beam = range(beam) if greedy else shuf_idx[i]
                for kk in cls_beam: #try until one is chemically valid
                    if success: break
                    clab, ilab = cls_topk[i][kk], icls_topk[i][kk]
                    node_feature = batch_idx.new_tensor( [clab, ilab] )
                    tree_batch.set_node_feature(new_node, node_feature)
                    
                    smiles, ismiles = self.vocab.get_smiles(clab), self.vocab.get_ismiles(ilab)
                    print(f"smiles : {smiles} and ismiles : {ismiles}")

                    
                    #  ^We need the primary motif obtained right after iron atom to contain :2 so that it can be attached to metal."
                    if primary_motif:
                        if ':2' not in ismiles:
                            continue
                        else :
                            new_atoms, new_bonds, attached, new_highlight_atoms, bonds_to_predict = graph_batch.add_primary_mol(bid, ismiles, root_indices_graph[bid])
                            new_bonds_expandlist[bid].extend(bonds_to_predict)
                            tree_batch.register_cgraph(new_node, new_atoms, new_bonds, attached, new_highlight_atoms)
                            success= True
                            continue
                    
                    fa_cluster, _, fa_used, fa_highlight = tree_batch.get_cluster(fa_node)
                    inter_cands, anchor_smiles, attach_points, complete_ligand = graph_batch.get_assm_cands(fa_cluster, fa_used, ismiles, fa_highlight)

                    # if primary_motif:
                    #     new_atoms, new_bonds, attached, highlight_atoms = graph_batch.add_primary_mol(bid, ismiles, root_indices_graph[bid])
                    #     tree_batch.register_cgraph(new_node, new_atoms, new_bonds, attached, highlight_atoms)
                    #     success= True
                    #     continue
                    

                    # ^ if the inter_cands is empty, then that means either all the atoms of the new cluster are :2 or there is a combination of :2 atoms and :0 atoms (basically, no :1). Now, we can still try to attach it to the parent cluster depending on whether there are compatible atoms in the parent cluster (enough :2 atoms) for it. 
                    if len(inter_cands) == 0:
                        if graph_batch.try_add_mol(bid, ismiles, [], fa_cluster, fa_used, fa_highlight):
                            new_atoms, new_bonds, attached, new_highlight_atoms, bonds_to_predict = graph_batch.add_mol(bid, ismiles, [], 0, root_indices_graph[bid], fa_cluster, fa_used, fa_highlight)
                            new_bonds_expandlist[bid].extend(bonds_to_predict)
                            tree_batch.register_cgraph(new_node, new_atoms, new_bonds, attached, new_highlight_atoms)
                            success = True
                            continue
                        else :
                            success = False
                            continue

                    elif len(inter_cands) == 1:
                        sorted_cands = [(inter_cands[0], 0)]
                        nth_child = 0

                    else:
                        nth_child = tree_batch.graph.in_degree(fa_node)
                        icls = [self.vocab[ (smiles,x) ][1] for x in anchor_smiles]
                        cands = inter_cands if len(attach_points) <= 2 else [ (x[0],x[-1]) for x in inter_cands ]
                        cand_vecs = self.enum_attach(hgraph, cands, icls, nth_child)

                        batch_idx = batch_idx.new_tensor( [bid] * len(inter_cands) )
                        assm_scores = self.get_assm_score(src_graph_vecs, batch_idx, cand_vecs).tolist()
                        sorted_cands = sorted( list(zip(inter_cands, assm_scores)), key = lambda x:x[1], reverse=True )

                    for inter_label,_ in sorted_cands:
                        inter_label = list(zip(inter_label, attach_points))
                        if graph_batch.try_add_mol(bid, ismiles, inter_label):
                            if complete_ligand:
                                new_atoms, new_bonds, attached, new_highlight_atoms, bonds_to_predict = graph_batch.add_mol(bid, ismiles, inter_label, nth_child, root_indices_graph[bid])
                                new_bonds_expandlist[bid].extend(bonds_to_predict)
                            else:
                                new_atoms, new_bonds, attached, new_highlight_atoms, bonds_to_predict = graph_batch.add_mol(bid, ismiles, inter_label, nth_child, root_indices_graph[bid], fa_cluster, fa_used, fa_highlight)
                                new_bonds_expandlist[bid].extend(bonds_to_predict)
                                
                            tree_batch.register_cgraph(new_node, new_atoms, new_bonds, attached, new_highlight_atoms)
                            tree_batch.update_attached(fa_node, inter_label) # ^ This is not modified for :2 atoms because it is possible for the :2 atom in parent motif to have more children attached to itself through another motif possibly. (Im not really sure, need to try with different motifs and check what output comes up.)
                            success = True
                            break

                if not success: #force backtrack
                    del new_motif_atom_pair[bid]
                    del new_motif_bond[bid]
                    child = stack[bid].pop() #pop the dummy new_node which can't be added
                    nth_child = tree_batch.graph.in_degree(stack[bid][-1]) 
                    edge_feature = batch_idx.new_tensor( [child, stack[bid][-1], nth_child, 0] ) # not adding distance here for the reverse edge between dummy node and parent node because it has not been established yet and thus it is not known.
                    new_edge = tree_batch.add_edge(child, stack[bid][-1], edge_feature)

                    child = stack[bid].pop() 
                    if len(stack[bid]) > 0:
                        nth_child = tree_batch.graph.in_degree(stack[bid][-1]) 
                        edge_distance = tree_batch.fmess[tree_batch.edge_dict[(stack[bid][-1], child)]][-1]
                        edge_feature = batch_idx.new_tensor( [child, stack[bid][-1], nth_child, edge_distance] )
                        new_edge = tree_batch.add_edge(child, stack[bid][-1], edge_feature)
            
            if any(new_motif_bond.values()):
                subtree_motif = subtree_motif[0], batch_idx.new_tensor(list(new_motif_bond.values())) 
                subgraph_motif = [], []
                htree_motif, hinter_motif, hgraph_motif = self.hmpn_motifdist(tree_tensors_motif, tree_tensors_motif, graph_tensors_motif, htree_motif, hinter_motif, hgraph_motif, subtree_motif, subgraph_motif)

                hmess_motif = self.rnn_cell_motif.get_hidden_state(htree_motif.mess)

                for bid, bond_idx in new_motif_bond.items():
                    edge_distance = self.W_dist_motifs(hmess_motif[bond_idx])
                                # for i in range(len(subtree[1])):
                    prev_feature = tree_batch.fmess[bond_idx]
                    new_feature = torch.cat([prev_feature[:3], edge_distance], dim=0)
                    tree_batch.fmess[bond_idx] = new_feature

            if any(new_bonds_expandlist.values()):
                atom_pairs = defaultdict(list)
                prediction_atoms = []
                for bid, bond_indices in new_bonds_expandlist.items():
                    for bond_idx in bond_indices:
                        atom_pair = graph_batch.get_atom_pair(bond_idx)
                        atom_pairs[bid].append(atom_pair)
                        prediction_atoms.append(atom_pair[0])
                        prediction_atoms.append(atom_pair[1])

                prediction_atoms = list(set(prediction_atoms)) # just to ensure there are no duplicates. 
                subgraph_atoms = batch_idx.new_tensor(prediction_atoms), batch_idx.new_tensor(new_bonds_expandlist)

                hgraph_atom = self.hmpn_distances(graph_tensors_atom, hgraph_atom, subgraph_atoms)
                hmess_atom = self.rnn_cell_atom.get_hidden_state(hgraph_atom.mess)

                for bid, bond_indices in new_bonds_expandlist.items():
                    for i, bond_idx in enumerate(bond_indices):
                        edge_distance = self.W_dist(hmess_atom[bond_idx])
                        graph_batch.predicted_distances[bid][atom_pairs[bid][i]] = edge_distance


            # create subtree and subgraph - message passing for new nodes and edges in the motif level and the graph level. 
            # predict distances for these new edges. 
            # update the fmess for these new edges with the distances.
        dirname = "metal_chelate_generated_complexes"
        return graph_batch.get_mol(dirname, root_indices_graph)

