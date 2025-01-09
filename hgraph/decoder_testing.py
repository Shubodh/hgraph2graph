import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from hgraph.nnutils import *
from hgraph.encoder_metal_dist import IncHierMPNEncoderMetalDist, IncHierMPNEncoderMetalDist_DistancePrediction
from hgraph.mol_graph_metal import MolGraphMetal

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
        self.rnn_cell = self.hmpn.tree_encoder.rnn
        self.rnn_cell_atom = self.hmpn_distances.graph_encoder.rnn
        self.E_assm = self.hmpn.E_i 
        # self.E_order = torch.eye(MolGraphMetal.MAX_POS).cuda()
        self.E_order = torch.eye(MolGraphMetal.MAX_POS)

        # Handles topological predictions, e.g., determining connectivity or structural relationships between nodes.
        # Input: Concatenated hidden and latent vectors.
        # Output: A scalar value (e.g., for binary classification).
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

        # self.W_dist=nn.Linear(hidden_size, 1) #! linear layer for distance prediction - try putting non-linearity here. 
        self.W_dist = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size),  # First layer (expansion)
                nn.ReLU(),

                nn.Linear(2 * hidden_size, 4*hidden_size),  # Second layer (compression)
                nn.ReLU(),

                nn.Linear(4*hidden_size, 8*hidden_size),
                nn.ReLU(),

                nn.Linear(8*hidden_size, 4*hidden_size),
                nn.ReLU(),

                nn.Linear(4*hidden_size, 2*hidden_size),
                nn.ReLU(),

                nn.Linear(2 * hidden_size, hidden_size),  # Second layer (compression)
                nn.ReLU(),

                nn.Linear(hidden_size, 1)  # Output layer (distance prediction)
        )
        self.W_atomdist = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size),  # First layer (expansion)
                nn.ReLU(),

                nn.Linear(2 * hidden_size, hidden_size),  # Second layer (compression)
                nn.ReLU(),

                nn.Linear(hidden_size, 1)  # Output layer (distance prediction)
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
        # #print("inside init_decoder_state")
        # #print('batch_size:',batch_size)
        # #print('src_root_vecs:',src_root_vecs.size())
        num_mess = len(tree_tensors[1])
        # #print('num_mess:',num_mess)
        agraph = tree_tensors[2].clone()
        bgraph = tree_tensors[3].clone()
        # #print('agraph:',agraph.size())
        # #print('bgraph:',bgraph.size())
        # #print('tree_tensors[-1]:',tree_tensors[-1])

        for i,tup in enumerate(tree_tensors[-1]):
            # #print("tup:",tup)
            root = tup[0]
            # #print('root:',root)
            # for node in tree_batch.nodes():
            #      #print('node:',node)
            # #print('agraph of root:',agraph[root])
            assert agraph[root,-1].item() == 0
            agraph[root,-1] = num_mess + i
            for v in tree_batch.successors(root):
                # #print('v:',v)
                mess_idx = tree_batch[root][v]['mess_idx'] 
                # #print(bgraph[mess_idx])
                assert bgraph[mess_idx,-1].item() == 0
                bgraph[mess_idx,-1] = num_mess + i

        new_tree_tensors = tree_tensors[:2] + [agraph, bgraph] + tree_tensors[4:]
        htree = HTuple()
        htree.mess = self.rnn_cell.get_init_state(tree_tensors[1], src_root_vecs)
        htree.emask = torch.cat( [bgraph.new_zeros(num_mess), bgraph.new_ones(batch_size)], dim=0 )
        # #print("initialized decoded state")

        return htree, new_tree_tensors

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
            icls_scores = self.iclsNN(cls_vecs) + vocab_masks #apply mask by log(x + mask): mask=0 or -INF
        return cls_scores, icls_scores

    def get_assm_score(self, src_graph_vecs, batch_idx, assm_vecs):
        if self.use_attention:
            assm_cxt = self.attention(src_graph_vecs, batch_idx, assm_vecs, self.A_assm)
        else:
            assm_cxt = index_select_ND(src_graph_vecs, 0, batch_idx)
        return (self.W_assm(assm_vecs) * assm_cxt).sum(dim=-1)

    def forward(self, src_mol_vecs, graphs, tensors, orders):

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
        # #print("---------------------------")
        # #print("batch_size = len(orders):",batch_size)
        tree_batch, graph_batch = graphs
        # #print("length of tree_batch, graph_batch:",len(tree_batch), len(graph_batch))
        tree_tensors, graph_tensors = tensors
        # #print("length of tree_tensors, graph_tensors:",len(tree_tensors), len(graph_tensors))
        # #print(graph_tensors[0].size())
        inter_tensors = tree_tensors
        true_distances = tree_tensors[1][:, 4]
        atom_distances=graph_tensors[1][:, 4]
        atom_hmess=graph_tensors[1]
        # #print(atom_hmess.shape)
        #print(len(atom_distances))
        #print(len(atom_hmess))
        # #print("true_distances:",len(true_distances))

        src_root_vecs, src_tree_vecs, src_graph_vecs = src_mol_vecs
        init_vecs = src_root_vecs if self.latent_size == self.hidden_size else self.W_root(src_root_vecs) # just a linear layer

        htree, tree_tensors = self.init_decoder_state(tree_batch, tree_tensors, init_vecs)
        hinter = HTuple(
            mess = self.rnn_cell.get_init_state(inter_tensors[1]),
            emask = self.itensor.new_zeros(inter_tensors[1].size(0))
        )
        hgraph = HTuple(
            mess = self.rnn_cell.get_init_state(graph_tensors[1]),
            vmask = self.itensor.new_zeros(graph_tensors[0].size(0)),
            emask = self.itensor.new_zeros(graph_tensors[1].size(0))
        )
        
        all_topo_preds, all_cls_preds, all_assm_preds = [], [], []
        new_atoms = []
        tree_scope = tree_tensors[-1]
        # print(graph_tensors[0])

        """
        root - it is assigned the root node from the tree_batch corresponding to the index stored in the tree_scope.

        clab - cluster index from hmap of pairVocab corresponding to the root node's smiles(motif). 
        ilab - tuple index from vmap of pairVocab corresponding to the root node's (smiles,ismiles).

        all_cls_preds - list of tuples (root_vec_encoded, batch_idx, clab, ilab) for each root node in the batch.

        new_atoms - list of atoms (flattened) belonging to this cluster(motif) with offset to account for the tree_batch corresponding to the root nodes of the batch.

        For our case, since the root node is the iron atom for every complex, clab and ilab will always be the same for all the root nodes. New atoms will only consist of the iron atom itself set by the graph(Atom) level offset. 

        The code is performing cluster predictions by mapping the labels of root nodes (from tree_scope) to cluster identifiers (clab and ilab) using a vocabulary (self.vocab). While it processes the same nodes from tree_scope, the purpose is to predict their cluster labels and prepare these predictions for further processing.
        """
        for i in range(batch_size):
            root = tree_batch.nodes[ tree_scope[i][0] ]
            clab, ilab = self.vocab[ root['label'] ] # label here is (smiles,ismiles) for the partiicular node. vocab maps it to a unique index for the cluster (clab) and another unique index for the tuple (ilab)
            all_cls_preds.append( (init_vecs[i], i, clab, ilab) ) #cluster prediction
            new_atoms.extend(root['cluster'])
        
        root_atoms = new_atoms.copy()
        # #print("root_atoms:",root_atoms)
        # print("new_atoms:",new_atoms)
        # #print("tree_scope:",tree_scope)     
        # exit()   
        # #print("new_atoms:",new_atoms)

        """
        subgraph

            INPUT: graph_batch (from tensorize), new_atoms (atoms belonging to the clusters of the root nodes - Iron everytime for our case), hgraph (The HTuple object initialized with mess, vmask, emask attributes)
            OUTPUT: tuple of (new_atom_index, new_bond_index) - tensors of the same dtype and device as vmask and emask respectively containing the indices of the new_atoms and the corresponding new_bonds between any of the atoms in the new_atoms list already existing in graph_batch. 
        
        graph_tensors

            uses embed_graph from the encoder to process the graph_tensors. It is then concatenated with the last element of the original graph_tensors which is just the scope of the graph_tensors.

        maxt - maximum length of the orders list in the batch.
        max_cls_size - maximum size of the clusters in the batch. It is calculated by taking the maximum of the length of the cluster of each node in the tree_batch and multiplying it by 2.

        """
        
        subgraph = self.update_graph_mask(graph_batch, new_atoms, hgraph)
        graph_tensors_atom=graph_tensors
        graph_tensors = self.hmpn.embed_graph(graph_tensors) + (graph_tensors[-1],) #preprocess graph tensors
        hgraph_atom=HTuple(
            mess = self.rnn_cell_atom.get_init_state(graph_tensors_atom[1]),
            vmask = self.itensor.new_zeros(graph_tensors_atom[0].size(0)),
            emask = self.itensor.new_zeros(graph_tensors_atom[1].size(0))
        )
        graph_tensors_atom = self.hmpn_distances.embed_graph(graph_tensors_atom) + (graph_tensors_atom[-1],) #preprocess graph tensors

        # #print("preprocessing done moving on to main loop")

        maxt = max([len(x) for x in orders])
        # #print("orders")
        # #print(len(orders[0]))
        # #print(len(orders[1]))
        # #print(len(orders[2]))
        # #print("maxt:",maxt)
        max_cls_size = max( [len(attr) * 2 for node,attr in tree_batch.nodes(data='cluster')] )
        # #print("max_cls_size:",max_cls_size)

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
        all_dist_preds_tree=[]
        # all_dist_preds_inter=[]
        # #print(orders[0])
        flagged_motifs = set()  # To store flagged edges for future checks
        flagged_edges = set()
        distance_preds_graph=[]
        distance_true_graph=torch.empty(0)

        edge_distances={}
        for edge in atom_hmess:
            key = (int(edge[0].item()), int(edge[1].item()))  # Keep direction (a, b)
            edge_distances[key] = edge[4]
        #print(len(edge_distances))
        #print(orders[0])

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

            subtree = htree.emask.new_tensor(subtree[0]), htree.emask.new_tensor(subtree[1]) 
            htree.emask.scatter_(0, subtree[1], 1)
            hinter.emask.scatter_(0, subtree[1], 1)

            cur_tree_tensors = self.apply_tree_mask(tree_tensors, htree, hgraph)
            cur_inter_tensors = self.apply_tree_mask(inter_tensors, hinter, hgraph)
            cur_graph_tensors = self.apply_graph_mask(graph_tensors, hgraph)
            # print("current tree tensors mess")
            # print(cur_tree_tensors[1])
            # print("current graph tensors mess")
            # print(cur_graph_tensors[1])
            htree, hinter, hgraph = self.hmpn(cur_tree_tensors, cur_inter_tensors, cur_graph_tensors, htree, hinter, hgraph, subtree, subgraph) 
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
            #! we can extract distances at the graph level.
            # ! normalization can maybe help to converge. If we can normalize the coordinates in a range of 0 to 1. 
            new_atoms = []
            prediction_atoms=[]
            # #print("htree.node:",htree.node[1])
            # exit()
            for i in batch_list:
                xid, yid, tlab = orders[i][t]
                #print("current dfs order: ", xid,yid)
                #print("flagged motifs",flagged_motifs)
                all_topo_preds.append( (htree.node[xid], i, tlab) ) #topology prediction
                if yid is not None:
                    mess_idx = tree_batch[xid][yid]['mess_idx']
                    new_atoms.extend( tree_batch.nodes[yid]['cluster'] ) #NOTE: regardless of tlab = 0 or 1
                    # distance prediction. 
                    hmess = self.rnn_cell.get_hidden_state(htree.mess)
                    e_dist=self.W_dist(hmess[mess_idx])
                    all_dist_preds_tree.append(e_dist)
                    #print("prediction for tree level motif", e_dist)
                    # exit()

                    # retrieving the graph edges from motifs. 
                    yid_atoms=tree_batch.nodes[yid]['cluster']
                    # #print(yid_atoms)
                    xid_atom=tree_batch.nodes[xid]['cluster']
                    # #print(xid_atom)
                    #print("xid atom", xid_atom)
                    #print("yid atom", yid_atoms)

                    # edges between root xid and yid_atoms
                    root_edges=[]
                    root_distances = []
                    if xid_atom[0] in root_atoms:
                        #print("there is a root atom")
                        # root_edges = [
                        #     edge[1] for edge in atom_hmess 
                        #     if edge[0] == xid_atom[0] and edge[1] in yid_atoms
                        # ]

                        for atom in yid_atoms:
                            fkey = (xid_atom[0], atom)  # Forward key
                            bkey = (atom, xid_atom[0])  # Backward key
                            
                            if fkey in edge_distances:
                                # root_edges.append(graph_batch[xid_atom[0]][atom]['mess_idx'])
                                root_edges.append(fkey)
                                root_distances.append(edge_distances[fkey])
                                prediction_atoms.append(xid_atom[0])
                                prediction_atoms.append(atom)
                                # root_edges.append(graph_batch[atom][xid_atom[0]]['mess_idx'])
                                # root_edges.append(bkey)
                                # root_distances.append(edge_distances[bkey])
                        #print("the edges present for this iteration connected to the root atom are: ", root_edges)
                        #print("the distances corresponding to these root edges are: ", root_distances)
                        
                        # #print("Length of root_edges ",len(root_edges))
                        # #print(root_edges)
                        # #print(len(root_distances))

                    #within yid_atoms
                    internal_edges=[]
                    internal_distances=[]
                    if yid not in flagged_motifs:
                        #print(f"{yid} is not in flagged motifs")
                        internal_edges = [
                            tuple(sorted((int(src.item()), int(tgt.item())))) for src, tgt, *_ in atom_hmess
                            if src in yid_atoms and tgt in yid_atoms and yid not in flagged_motifs
                        ]
                        internal_edges = list(set(internal_edges))
                        for edge in internal_edges:
                            prediction_atoms.append(edge[0])
                            prediction_atoms.append(edge[1])
                            internal_distances.append(edge_distances[edge])
                        #print("the internal edges of yid motif are : ",internal_edges)
                        #print("the internal distances of these yid edges are :", internal_distances)

                    # Flag them to avoid reprocessing while backtracking in dfs. Only necessary for edges within yid_atoms
                    flagged_motifs.add(yid)

                    # now we update the hgraph_atom for the prediction atoms.
                    subgraph_atoms=self.update_graph_mask(graph_batch,prediction_atoms,hgraph_atom)
                    prediction_graph_tensors=self.apply_graph_mask(graph_tensors_atom,hgraph_atom)
                    hgraph_atom=self.hmpn_distances(graph_tensors_atom, hgraph_atom, subgraph)
                    
                    # #print(root_edges)
                    # for j,mess_idx in enumerate(root_edges):
                    #print("Prediction and true distance for root edges")
                    for j,key in enumerate(root_edges):
                        # distance prediction. 
                        mess_idx=graph_batch[key[0]][key[1]]['mess_idx']
                        hmess=self.rnn_cell_atom.get_hidden_state(hgraph.mess)
                        # print(hmess)
                        # print(hgraph.mess)
                        # exit()
                        e_dist=self.W_dist(hmess[mess_idx])
                        distance_preds_graph.append(e_dist)
                        # distance_preds_graph.append(e_dist)
                        # distance_true_graph.append(root_distances[j])
                        distance_true_graph = torch.cat([distance_true_graph, edge_distances[key].unsqueeze(0)])

                        #print(f"prediction is {e_dist} and true is {edge_distances[key].unsqueeze(0)}")

                    #print("Prediction and true distance for internal edges")
                    for j,edge in enumerate(internal_edges):
                        mess_idx=graph_batch[edge[0]][edge[1]]['mess_idx']
                        hmess=self.rnn_cell_atom.get_hidden_state(hgraph.mess)
                        e_dist=self.W_dist(hmess[mess_idx])
                        # distance_preds_graph.append(e_dist)
                        distance_preds_graph.append(e_dist)
                        # distance_true_graph.append(internal_distances[j])
                        distance_true_graph = torch.cat([distance_true_graph, edge_distances[edge].unsqueeze(0)])

                        #print(f"prediction is {e_dist} and true is {edge_distances[edge].unsqueeze(0)}")
                # #print(distance_preds_graph)
                # #print(distance_true_graph)
                # exit()

                                    
                if tlab == 0: continue

                cls = tree_batch.nodes[yid]['smiles']
                clab, ilab = self.vocab[ tree_batch.nodes[yid]['label'] ]
                mess_idx = tree_batch[xid][yid]['mess_idx']
                hmess = self.rnn_cell.get_hidden_state(htree.mess)
                all_cls_preds.append( (hmess[mess_idx], i, clab, ilab) ) #cluster prediction using message
                
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

        # distance_preds_graph=torch.stack(distance_preds_graph).squeeze()
        # distance_true_graph=torch.stack(distance_true_graph).squeeze()
        # #print(torch.stack(distance_true_graph))
        #print((distance_true_graph.shape))
        #print(len(distance_preds_graph))
        predicted_tensor = torch.cat(distance_preds_graph, dim=0)
        assert predicted_tensor.shape == distance_true_graph.shape, f"Shape mismatch: {predicted_tensor.shape} vs {distance_true_graph.shape}"
        distloss = self.dist_loss(predicted_tensor, distance_true_graph)
        # dist_loss_atoms=self.dist_loss(torch.stack(distance_preds_graph).squeeze(),torch.stack(distance_true_graph).squeeze())

        # dist_loss=self.dist_loss(torch.cat(all_dist_preds_tree,dim=0),true_distances[1:])
        # exit()

        # print(true_distances[1:])
        # exit()


        if len(all_assm_preds) > 0:
            assm_vecs, batch_idx, assm_labels = zip_tensors_metal(all_assm_preds)
            assm_scores = self.get_assm_score(src_graph_vecs, batch_idx, assm_vecs)
            assm_loss = self.assm_loss(assm_scores, assm_labels)
            assm_acc = get_accuracy_sym(assm_scores, assm_labels)
        else:
            assm_loss, assm_acc = 0, 1
        
        loss = (topo_loss + cls_loss + assm_loss + distloss) / batch_size
        return loss, cls_acc, icls_acc, topo_acc, assm_acc, distloss

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

    # def decode(self, src_mol_vecs, greedy=True, max_decode_step=100, beam=5):
    #     src_root_vecs, src_tree_vecs, src_graph_vecs = src_mol_vecs
    #     batch_size = len(src_root_vecs)

    #     tree_batch = IncTree(batch_size, node_fdim=2, edge_fdim=3)
    #     graph_batch = IncGraph(self.avocab, batch_size, node_fdim=self.hmpn.atom_size, edge_fdim=self.hmpn.atom_size + self.hmpn.bond_size)
    #     stack = [[] for i in range(batch_size)]

    #     init_vecs = src_root_vecs if self.latent_size == self.hidden_size else self.W_root(src_root_vecs)
    #     batch_idx = self.itensor.new_tensor(range(batch_size))
    #     cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, batch_idx, init_vecs, None)
    #     root_cls = cls_scores.max(dim=-1)[1]
    #     icls_scores = icls_scores + self.vocab.get_mask(root_cls)
    #     root_cls, root_icls = root_cls.tolist(), icls_scores.max(dim=-1)[1].tolist()

    #     super_root = tree_batch.add_node() 
    #     for bid in range(batch_size):
    #         clab, ilab = root_cls[bid], root_icls[bid]
    #         root_idx = tree_batch.add_node( batch_idx.new_tensor([clab, ilab]) )
    #         tree_batch.add_edge(super_root, root_idx) 
    #         stack[bid].append(root_idx)

    #         root_smiles = self.vocab.get_ismiles(ilab)
    #         new_atoms, new_bonds, attached = graph_batch.add_mol(bid, root_smiles, [], 0)
    #         tree_batch.register_cgraph(root_idx, new_atoms, new_bonds, attached)
        
    #     #invariance: tree_tensors is equal to inter_tensors (but inter_tensor's init_vec is 0)
    #     tree_tensors = tree_batch.get_tensors()
    #     graph_tensors = graph_batch.get_tensors()

    #     htree = HTuple( mess = self.rnn_cell.get_init_state(tree_tensors[1]) )
    #     hinter = HTuple( mess = self.rnn_cell.get_init_state(tree_tensors[1]) )
    #     hgraph = HTuple( mess = self.rnn_cell.get_init_state(graph_tensors[1]) )
    #     h = self.rnn_cell.get_hidden_state(htree.mess)
    #     h[1 : batch_size + 1] = init_vecs #wiring root (only for tree, not inter)
        
    #     for t in range(max_decode_step):
    #         batch_list = [ bid for bid in range(batch_size) if len(stack[bid]) > 0 ]
    #         if len(batch_list) == 0: break

    #         batch_idx = batch_idx.new_tensor(batch_list)
    #         cur_tree_nodes = [stack[bid][-1] for bid in batch_list]
    #         subtree = batch_idx.new_tensor(cur_tree_nodes), batch_idx.new_tensor([])
    #         subgraph = batch_idx.new_tensor( tree_batch.get_cluster_nodes(cur_tree_nodes) ), batch_idx.new_tensor( tree_batch.get_cluster_edges(cur_tree_nodes) )

    #         htree, hinter, hgraph = self.hmpn(tree_tensors, tree_tensors, graph_tensors, htree, hinter, hgraph, subtree, subgraph)
    #         topo_scores = self.get_topo_score(src_tree_vecs, batch_idx, htree.node.index_select(0, subtree[0]))
    #         topo_scores = torch.sigmoid(topo_scores)
    #         if greedy:
    #             topo_preds = topo_scores.tolist()
    #         else:
    #             topo_preds = torch.bernoulli(topo_scores).tolist()

    #         new_mess = []
    #         expand_list = []
    #         for i,bid in enumerate(batch_list):
    #             if topo_preds[i] > 0.5 and tree_batch.can_expand(stack[bid][-1]):
    #                 expand_list.append( (len(new_mess), bid) )
    #                 new_node = tree_batch.add_node() #new node label is yet to be predicted
    #                 edge_feature = batch_idx.new_tensor( [stack[bid][-1], new_node, 0] ) #parent to child is 0
    #                 new_edge = tree_batch.add_edge(stack[bid][-1], new_node, edge_feature) 
    #                 stack[bid].append(new_node)
    #                 new_mess.append(new_edge)
    #             else:
    #                 child = stack[bid].pop()
    #                 if len(stack[bid]) > 0:
    #                     nth_child = tree_batch.graph.in_degree(stack[bid][-1]) #edge child -> father has not established
    #                     edge_feature = batch_idx.new_tensor( [child, stack[bid][-1], nth_child] )
    #                     new_edge = tree_batch.add_edge(child, stack[bid][-1], edge_feature)
    #                     new_mess.append(new_edge)

    #         subtree = subtree[0], batch_idx.new_tensor(new_mess)
    #         subgraph = [], []
    #         htree, hinter, hgraph = self.hmpn(tree_tensors, tree_tensors, graph_tensors, htree, hinter, hgraph, subtree, subgraph)
    #         cur_mess = self.rnn_cell.get_hidden_state(htree.mess).index_select(0, subtree[1])

    #         if len(expand_list) > 0:
    #             idx_in_mess, expand_list = zip(*expand_list)
    #             idx_in_mess = batch_idx.new_tensor( idx_in_mess )
    #             expand_idx = batch_idx.new_tensor( expand_list )
    #             forward_mess = cur_mess.index_select(0, idx_in_mess)
    #             cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, expand_idx, forward_mess, None)
    #             scores, cls_topk, icls_topk = hier_topk(cls_scores, icls_scores, self.vocab, beam)
    #             if not greedy:
    #                 scores = torch.exp(scores) #score is output of log_softmax
    #                 shuf_idx = torch.multinomial(scores, beam, replacement=True).tolist()

    #         for i,bid in enumerate(expand_list):
    #             new_node, fa_node = stack[bid][-1], stack[bid][-2]
    #             success = False
    #             cls_beam = range(beam) if greedy else shuf_idx[i]
    #             for kk in cls_beam: #try until one is chemically valid
    #                 if success: break
    #                 clab, ilab = cls_topk[i][kk], icls_topk[i][kk]
    #                 node_feature = batch_idx.new_tensor( [clab, ilab] )
    #                 tree_batch.set_node_feature(new_node, node_feature)
    #                 smiles, ismiles = self.vocab.get_smiles(clab), self.vocab.get_ismiles(ilab)
    #                 fa_cluster, _, fa_used = tree_batch.get_cluster(fa_node)
    #                 inter_cands, anchor_smiles, attach_points = graph_batch.get_assm_cands(fa_cluster, fa_used, ismiles)

    #                 if len(inter_cands) == 0:
    #                     continue
    #                 elif len(inter_cands) == 1:
    #                     sorted_cands = [(inter_cands[0], 0)]
    #                     nth_child = 0
    #                 else:
    #                     nth_child = tree_batch.graph.in_degree(fa_node)
    #                     icls = [self.vocab[ (smiles,x) ][1] for x in anchor_smiles]
    #                     cands = inter_cands if len(attach_points) <= 2 else [ (x[0],x[-1]) for x in inter_cands ]
    #                     cand_vecs = self.enum_attach(hgraph, cands, icls, nth_child)

    #                     batch_idx = batch_idx.new_tensor( [bid] * len(inter_cands) )
    #                     assm_scores = self.get_assm_score(src_graph_vecs, batch_idx, cand_vecs).tolist()
    #                     sorted_cands = sorted( list(zip(inter_cands, assm_scores)), key = lambda x:x[1], reverse=True )

    #                 for inter_label,_ in sorted_cands:
    #                     inter_label = list(zip(inter_label, attach_points))
    #                     if graph_batch.try_add_mol(bid, ismiles, inter_label):
    #                         new_atoms, new_bonds, attached = graph_batch.add_mol(bid, ismiles, inter_label, nth_child)
    #                         tree_batch.register_cgraph(new_node, new_atoms, new_bonds, attached)
    #                         tree_batch.update_attached(fa_node, inter_label)
    #                         success = True
    #                         break

    #             if not success: #force backtrack
    #                 child = stack[bid].pop() #pop the dummy new_node which can't be added
    #                 nth_child = tree_batch.graph.in_degree(stack[bid][-1]) 
    #                 edge_feature = batch_idx.new_tensor( [child, stack[bid][-1], nth_child] )
    #                 new_edge = tree_batch.add_edge(child, stack[bid][-1], edge_feature)

    #                 child = stack[bid].pop() 
    #                 if len(stack[bid]) > 0:
    #                     nth_child = tree_batch.graph.in_degree(stack[bid][-1]) 
    #                     edge_feature = batch_idx.new_tensor( [child, stack[bid][-1], nth_child] )
    #                     new_edge = tree_batch.add_edge(child, stack[bid][-1], edge_feature)

    #     return graph_batch.get_mol()


