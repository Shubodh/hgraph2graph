import torch
import rdkit.Chem as Chem
import networkx as nx
from hgraph.mol_graph_metal import MolGraphMetal
from hgraph.chemutils import *
from collections import defaultdict
import os

class IncBaseMetal(object):
    """
    The IncBase class implements an incremental graph construction utility for managing directed graphs, node features, and edge features.    
    """

    def __init__(self, batch_size, node_fdim, edge_fdim, max_nodes=100, max_edges=200, max_nb=12):
        """
        max_nb: maximum number of neighbors for each node
        max_nodes: maximum number of nodes in the graph
        max_edges: maximum number of edges in the graph
        node_fdim: dimension of node features
        edge_fdim: dimension of edge features
        """
        self.max_nb = max_nb
        self.graph = nx.DiGraph()
        self.graph.add_node(0) #make sure node is 1 index
        self.edge_dict = {None : 0} #make sure edge is 1 index
        self.edge_atom_dict = {None : 0} #make sure edge is 1 index

        # self.fnode = torch.zeros(max_nodes * batch_size, node_fdim).long().cuda()
        self.fnode = torch.zeros(max_nodes * batch_size, node_fdim).long()
        self.fmess = self.fnode.new_zeros(max_edges * batch_size, edge_fdim)
        self.agraph = self.fnode.new_zeros(max_edges * batch_size, max_nb)
        self.bgraph = self.fnode.new_zeros(max_edges * batch_size, max_nb)

    def add_node(self, feature=None):
        """
        Add a node to the graph with the given feature.
        """
        idx = len(self.graph)
        self.graph.add_node(idx)
        if feature is not None:
            self.fnode[idx, :len(feature)] = feature
        return idx

    def set_node_feature(self, idx, feature):
        self.fnode[idx, :len(feature)] = feature

    def can_expand(self, idx):
        """
        Checks if the node can have more edges/neighbors.
        """
        return self.graph.in_degree(idx) < self.max_nb

    def add_edge(self, i, j, feature=None):
        """
        adds a directed edge from i to j and updates the edge feature.
        updates agraph of the j node with this new edge index.
        updates bgraph of this edge with the incoming edges to i (excluding from j).

        updates the bgraph of the successor edges of j (excluding to i).
        """
        if (i,j) in self.edge_dict: 
            return self.edge_dict[(i,j)]

        self.graph.add_edge(i, j)
        self.edge_dict[(i,j)] = idx = len(self.edge_dict)
        self.edge_atom_dict[idx] = (i,j)

        self.agraph[j, self.graph.in_degree(j) - 1] = idx
        if feature is not None:
            self.fmess[idx, :len(feature)] = feature

        in_edges = [self.edge_dict[(k,i)] for k in self.graph.predecessors(i) if k != j]
        self.bgraph[idx, :len(in_edges)] = self.fnode.new_tensor(in_edges)

        for k in self.graph.successors(j):
            if k == i: continue
            nei_idx = self.edge_dict[(j,k)]
            self.bgraph[nei_idx, self.graph.in_degree(j) - 2] = idx

        return idx
    
    def get_atom_pair(self, bond_idx):
        """
        Get the atom pair corresponding to the bond index. This will return a tuple of the atom-pair. 
        """
        if bond_idx in self.edge_atom_dict:
            return self.edge_atom_dict[bond_idx]
        else:
            raise ValueError(f"Bond index {bond_idx} not found in edge_atom_dict.")


class IncTreeMetal(IncBaseMetal):

    def __init__(self, batch_size, node_fdim, edge_fdim, max_nodes=100, max_edges=200, max_nb=12, max_sub_nodes=20):
        super(IncTreeMetal, self).__init__(batch_size, node_fdim, edge_fdim, max_nodes, max_edges, max_nb)
        self.cgraph = self.fnode.new_zeros(max_nodes * batch_size, max_sub_nodes)

    def get_tensors(self):
        return self.fnode, self.fmess, self.agraph, self.bgraph, self.cgraph, None 

    def register_cgraph(self, i, nodes, edges, attached, highlight_atoms):
        self.cgraph[i, :len(nodes)] = self.fnode.new_tensor(nodes)
        self.graph.nodes[i]['cluster'] = nodes
        self.graph.nodes[i]['cluster_edges'] = edges
        self.graph.nodes[i]['attached'] = attached
        self.graph.nodes[i]['highlight_atoms'] = highlight_atoms

    def update_attached(self, i, attached):
        if len(self.graph.nodes[i]['cluster']) > 1: 
            used = list(zip(*attached))[0]
            self.graph.nodes[i]['attached'].extend(used)
    
    # def update_edges(self, i, edges):
    #     self.graph.nodes[i]['cluster_edges'].extend(edges)

    def get_cluster(self, node_idx):
        cluster = self.graph.nodes[node_idx]['cluster']
        edges = self.graph.nodes[node_idx]['cluster_edges']
        used = self.graph.nodes[node_idx]['attached']
        highlight_atoms = self.graph.nodes[node_idx]['highlight_atoms']
        return cluster, edges, used

    def get_cluster_nodes(self, node_list):
        return [ c for node_idx in node_list for c in self.graph.nodes[node_idx]['cluster'] ]

    def get_cluster_edges(self, node_list):
        return [ e for node_idx in node_list for e in self.graph.nodes[node_idx]['cluster_edges'] ]


class IncGraphMetal(IncBaseMetal):

    def __init__(self, avocab, batch_size, node_fdim, edge_fdim, max_nodes=100, max_edges=300, max_nb=10):
        super(IncGraphMetal, self).__init__(batch_size, node_fdim, edge_fdim, max_nodes, max_edges, max_nb)
        self.avocab = avocab
        self.mol = Chem.RWMol()
        self.mol.AddAtom( Chem.Atom('C') ) #make sure node is 1 index, consistent to self.graph
        self.fnode = self.fnode.float()
        self.fmess = self.fmess.float()
        self.batch = defaultdict(list)
        self.batch_highlights = defaultdict(list)
        self.predicted_distances = {}
        self.mol_bonds = {}
    
    def connect_ligand(self, bid, tree_idx, graph_idx):
        new_bonds = []
        highlight_atoms = list(set(self.batch_highlights[bid]))
        for atom_idx in highlight_atoms:
            existing_bond = self.mol_bonds.get((graph_idx, atom_idx), None)
            if existing_bond is None:
                self.mol_bonds[(graph_idx, atom_idx)] = self.mol_bonds[(atom_idx, graph_idx)] = 4 
                self.add_edge(graph_idx, atom_idx, self.get_mess_feature(self.mol.GetAtomWithIdx(graph_idx), 4, 0) )
                self.add_edge(atom_idx, graph_idx, self.get_mess_feature(self.mol.GetAtomWithIdx(atom_idx), 4, 0) )
                new_bonds.extend( [ self.edge_dict[(graph_idx, atom_idx)], self.edge_dict[(atom_idx, graph_idx)] ] )
        
        # return new_bonds

    def get_mol(self, dirname, root_indices_graph):
        if not os.path.exists(dirname):
                os.makedirs(dirname)

        for batch_idx, batch_atoms in self.batch.items():
            n_atoms = len(batch_atoms)

            atom_mapping = {atom_id: i for i, atom_id in enumerate(batch_atoms)}

            bond_dist_dict = self.predicted_distances[batch_idx]  # {(a1, a2): distance}
            edge_list = []
            for (a1, a2), dist in bond_dist_dict.items():
                edge_list.append((atom_mapping[a1], atom_mapping[a2], dist))
            
            atom_labels = []
            root_atom = 0
            for idx in batch_atoms:
                atom = self.mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == '[Fe]':
                    atom_labels.append('Fe')
                else:
                    atom_labels.append(atom.GetSymbol())  # or use any other label you want
                if idx in root_indices_graph:
                    root_atom = atom_mapping[idx]

            recover_coordinates(
                n_atoms=n_atoms,
                edge_list=edge_list,
                root_atom=root_atom,
                export_xyz_path=f"{dirname}/mol_{batch_idx}.xyz",  # Optional export
                atom_labels=atom_labels
            )

    def get_tensors(self):
        return self.fnode, self.fmess, self.agraph, self.bgraph, None 
    
    def add_primary_mol(self, batch_idx, smiles, root_atom_index):
        emol = get_mol(smiles)
        new_atoms, new_bonds, attached, highlight_atoms = [], [], [], []
        bonds_for_prediciton = []
        for atom in emol.GetAtoms():
            new_atom = copy_atom(atom)
            new_atom.SetAtomMapNum( batch_idx ) 
            idx = self.mol.AddAtom( new_atom )
            assert idx == self.add_node( self.get_atom_feature(new_atom) )
            new_atoms.append(idx)
            self.batch[batch_idx].append(idx)
            # We do not add any attached to the node of the primary node. This isbecause in mol_graph_metal.py, we are renumbering the molecules such that highlight_atoms are starting with the index zero. and in inter_label within label_tree, originally :1 was added to the root node when there were no inter_atoms, but in this case, we have :2 for the root node due to the renumbering of the indices. Since this is the case, :1 present in this primary motif can be a possible attachment point for its future atoms in the next motif. 
            if atom.GetAtomMapNum() == 2 : # ^ Attach only those atoms to the metal center which have :2 in them. 
                highlight_atoms.append(idx) # ^ Added this to keep track of the atoms which have :2 in them.
                self.batch_highlights[batch_idx].append(idx)
                existing_bond = self.mol_bonds.get((root_atom_index, idx), None)
                if existing_bond is None:
                    self.mol_bonds[(root_atom_index, idx)] = self.mol_bonds[(idx, root_atom_index)] = 4 
                    self.add_edge(root_atom_index, idx, self.get_mess_feature(self.mol.GetAtomWithIdx(root_atom_index), 4, 0) ) # 4 is just for the representation of the new type of bond. 
                    self.add_edge(idx, root_atom_index, self.get_mess_feature(self.mol.GetAtomWithIdx(idx), 4, 0) )
                    bonds_for_prediciton.append(self.edge_dict[(root_atom_index, idx)])
        
        return new_atoms, new_bonds, attached, highlight_atoms, bonds_for_prediciton


    def add_mol(self, batch_idx, smiles, inter_label, nth_child, root_atom_index = None, fa_cluster = None, fa_used = None, fa_highlight = None):
        emol = get_mol(smiles)


        # ^ new code 
        atom_map_highlight = {}
        if fa_cluster is not None :
            # new_highlight_atoms = []
            emol = get_mol(smiles)
            highlight_atoms_parent = fa_highlight.copy()
            highlight_atoms_emol = [atom.GetIdx() for atom in emol.GetAtoms() if atom.GetAtomMapNum() == 2]

            unmatched_parents = set(highlight_atoms_parent)
            
            if highlight_atoms_emol: # this should be true by default, but just keeping it for a sanity check. 
                for highlight_atom in highlight_atoms_emol:
                    for parent_atom in list(unmatched_parents):  # iterate over remaining unmatched ones
                        if parent_atom not in fa_used and atom_equal(emol.GetAtomWithIdx(highlight_atom), self.mol.GetAtomWithIdx(parent_atom)):
                            unmatched_parents.remove(parent_atom)  # mark this parent as used
                            atom_map_highlight[highlight_atom] = parent_atom
                            break

        atom_map = {y : x for x,y in inter_label}
        atom_map.update(atom_map_highlight) #^ Added this to club :2 atoms as well.
        new_atoms, new_bonds, attached = [], [], []
        highlight_atoms = [] 
        bonds_for_prediction = []

        for atom in emol.GetAtoms(): #atoms must be inserted in order given by emol.GetAtoms() (for rings assembly)
            if atom.GetIdx() in atom_map: 
                idx = atom_map[atom.GetIdx()]
                new_atoms.append(idx)
                attached.append(idx) # ^ this will also add those :2 atoms which have been dissolved with its parent motifs atoms into attached. There cant be 2-hop attached for these atoms. 
                if atom.GetAtomMapNum() == 2: 
                    highlight_atoms.append(idx) #^ Added this to keep track of the atoms which have :2 in them.
                    self.batch_highlights[batch_idx].append(idx) # ! I think this is repetitive and we can remove this since the parent cluster atoms are already in the highlight_atoms. but anyways while attaching these to iron, will convert it into a set to avoid repetitions. 
            else:
                new_atom = copy_atom(atom)
                new_atom.SetAtomMapNum( batch_idx ) 
                idx = self.mol.AddAtom( new_atom )
                assert idx == self.add_node( self.get_atom_feature(new_atom) ) #mol and nx graph must have the same indexing
                atom_map[atom.GetIdx()] = idx
                new_atoms.append(idx)
                self.batch[batch_idx].append(idx)
                if atom.GetAtomMapNum() == 1: attached.append(idx)
                elif atom.GetAtomMapNum() == 2 : 
                    highlight_atoms.append(idx) #!
                    self.batch_highlights[batch_idx].append(idx)

        for bond in emol.GetBonds():
            a1 = atom_map[bond.GetBeginAtom().GetIdx()]
            a2 = atom_map[bond.GetEndAtom().GetIdx()]
            if a1 == a2: continue
            bond_type = bond.GetBondType()
            existing_bond = self.mol_bonds.get((a1, a2), None) # !

            if existing_bond is None:
                self.mol_bonds[(a1, a2)] = self.mol_bonds[(a2, a1)] = bond_type #!
                
                self.add_edge(a1, a2, self.get_mess_feature(bond.GetBeginAtom(), bond_type, nth_child if a2 in attached else 0) ) #only child to father node (in intersection) have non-zero nth_child
                self.add_edge(a2, a1, self.get_mess_feature(bond.GetEndAtom(), bond_type, nth_child if a1 in attached else 0) ) 

                bonds_for_prediction.append(self.edge_dict[(a1,a2)])

            else:
                attached.extend( [(a1,a2),(a2,a1)] )
            new_bonds.extend( [ self.edge_dict[(a1,a2)], self.edge_dict[(a2,a1)] ] )
        
        # ! New code for adding atom level bond between metal centre and ligand.
        # if root_atom_index is not None:
        #     for atom_idx in highlight_atoms:
        #         existing_bond = self.mol_bonds.get((root_atom_index, atom_idx), None)
        #         if existing_bond is None:
        #             self.mol_bonds[(root_atom_index, atom_idx)] = self.mol_bonds[(atom_idx, root_atom_index)] = 4 
        #             self.add_edge(root_atom_index, atom_idx, self.get_mess_feature(self.mol.GetAtomWithIdx(root_atom_index), 4, 0) ) # 4 is just for the representation of the new type of bond. 
        #             self.add_edge(atom_idx, root_atom_index, self.get_mess_feature(self.mol.GetAtomWithIdx(atom_idx), 4, 0) )

        if emol.GetNumAtoms() == 1: #singletons always attached = []
            attached = []
        return new_atoms, new_bonds, attached, highlight_atoms, bonds_for_prediction

    #validity check function
    def try_add_mol(self, batch_idx, smiles, inter_label, fa_cluster = None, fa_used = None, fa_highlight = None):
        
        """
        Okay, so what im doing in the below code snippet is the following. 

        I have fa_highlight which are the parent atoms in the parent cluster and i have extracted :2 atoms in the new possible cluster. Now, i am trying to match the :2 atoms in the new cluster with the :2 atoms in the parent cluster. If there is atleast one match, then i will add the :2 atom in the new cluster to the atom_map_highlight dictionary. This is done to ensure that a new atom is not created for already matched atoms with its parent cluster. 

        Right now, i am assuming that it is enough to have only one :2 attachment to the new parent cluster. But, we will need to see how it works out in practice. 
        """
        atom_map_highlight = {}
        if fa_cluster is not None :
            # new_highlight_atoms = []
            emol = get_mol(smiles)
            highlight_atoms_parent = fa_highlight.copy()
            highlight_atoms_emol = [atom.GetIdx() for atom in emol.GetAtoms() if atom.GetAtomMapNum() == 2]

            unmatched_parents = set(highlight_atoms_parent)
            
            if highlight_atoms_emol: # this should be true by default, but just keeping it for a sanity check. 
                matched = False
                for highlight_atom in highlight_atoms_emol:
                    for parent_atom in list(unmatched_parents):  # iterate over remaining unmatched ones
                        if parent_atom not in fa_used and atom_equal(emol.GetAtomWithIdx(highlight_atom), self.mol.GetAtomWithIdx(parent_atom)):
                            unmatched_parents.remove(parent_atom)  # mark this parent as used
                            atom_map_highlight[highlight_atom] = parent_atom
                            matched=True
                            break
            
            # Not considering the motif if there is not atlease a single match between the :2 atoms in the new cluster and the parent cluster.
            if not matched:
                return False
            
            
            # for highlight_atom in highlight_atoms_emol:
            #     if highlight_atom not in atom_map_highlight:
            #         new_highlight_atoms.append(highlight_atom)        
                

                    
        # Previous code. 
        emol = get_mol(smiles)
        for x,y in inter_label:
            if not atom_equal(self.mol.GetAtomWithIdx(x), emol.GetAtomWithIdx(y)):
                return False

        atom_map = {y : x for x,y in inter_label}
        new_atoms, new_bonds = [], []
        atom_map.update(atom_map_highlight) #^ Added this to club :2 atoms as well. 

        for atom in emol.GetAtoms(): #atoms must be inserted in order given by emol.GetAtoms() (for rings assembly)
            if atom.GetIdx() not in atom_map: 
                new_atom = copy_atom(atom)
                new_atom.SetAtomMapNum( batch_idx ) 
                idx = self.mol.AddAtom( new_atom )
                atom_map[atom.GetIdx()] = idx
                new_atoms.append(idx)

        valid = True
        for bond in emol.GetBonds():
            a1 = atom_map[bond.GetBeginAtom().GetIdx()]
            a2 = atom_map[bond.GetEndAtom().GetIdx()]
            if a1 == a2: #self loop must be an error
                valid = False
                break
            # ! commenting this for now since we are not checking for tmp_mol currently.
            # bond_type = bond.GetBondType()
            # existing_bond = self.mol_bonds.get((a1, a2), None) # ^ modified since not adding actual bonds in the molecule. 
            # if existing_bond is None: #later maybe check bond type match
            #     self.mol_bonds[(a1, a2)] = bond_type #!
            #     new_bonds.append( (a1,a2) )

        # ! commenting this part because it is creating a sub molecule with the existing atoms in the molecule and the new atoms. We cannot do this because we have asigned a number 4 to the bond type between iron and atoms. We dont actually have a rdkit bond_type between them. Maybe we can go ahead and add dative_bond_type? 
        # if valid: 
        #     tmp_mol = get_sub_mol_metal(self.mol, self.batch[batch_idx] + new_atoms, self.mol_bonds)
        #     tmp_mol = sanitize(tmp_mol, kekulize=False)
        
        #revert trial
        # ! commenting this for now since we are not checking for tmp_mol currently. 
        # for a1,a2 in new_bonds:
        #     del self.mol_bonds[(a1, a2)]
        for atom in sorted(new_atoms, reverse=True): 
            self.mol.RemoveAtom(atom)

        # return valid and (tmp_mol is not None)
        return valid

    def get_atom_feature(self, atom):
        """
        This returns a one-hot tensor of the atom symbol and formal charge.
        """
        f = torch.zeros(self.avocab.size())
        symbol, charge = atom.GetSymbol(), atom.GetFormalCharge()
        f[ self.avocab[(symbol,charge)] ] = 1
        # return f.cuda()
        return f

    def get_mess_feature(self, atom, bond_type, nth_child): # ^ Modified this function to make a new column for new type of bond between metal and atom since that is not explicitly present in MolGraphMetal.BOND_LIST.
        """
        returns a concatenated tensor of the atom feature, bond feature, and nth_child feature.
        """
        f1 = torch.zeros(self.avocab.size())
        f2 = torch.zeros(len(MolGraphMetal.BOND_LIST)+1) # added +1 for the bond between metal centre and ligand
        f3 = torch.zeros(MolGraphMetal.MAX_POS)
        symbol, charge = atom.GetSymbol(), atom.GetFormalCharge()
        f1[ self.avocab[(symbol,charge)] ] = 1
        try:
            f2[ MolGraphMetal.BOND_LIST.index(bond_type) ] = 1
        except:
            f2[ -1 ] = 1
        f3[ nth_child ] = 1
        # return torch.cat( [f1,f2,f3], dim=-1 ).cuda()
        return torch.cat( [f1,f2,f3], dim=-1 )


    def get_assm_cands(self, cluster, used, smiles, highlight_atoms):
        # ! Need to modify this in a way to generate possible attachment points in the parent cluster for :2 atoms in the new ismiles. 
        """
        keeping everything the same, except for the condition in attachment point which has been changed to the requirement of the atom map number to be 1 instead of >0 to account for :2.
        """
        emol = get_mol(smiles)
        if emol.GetNumAtoms() == 1:
            attach_points = [0]
        else:
            attach_points = [atom.GetIdx() for atom in emol.GetAtoms() if atom.GetAtomMapNum() == 1] #^ Changed this to only account for :1 atoms while generating attach_points. 
            metal_points = [atom.GetIdx() for atom in emol.GetAtoms() if atom.GetAtomMapNum() == 2] #^ Added this to keep track of which atom indices in the new cluster have :2

        inter_size = len(attach_points)
        idxfunc = lambda x:x.GetIdx()
        anchors = attach_points

        if inter_size == 0:
            if len(metal_points)>0:
                return [], [], [], False
            else:
                return [], [], [], True
        if inter_size == 1:
            anchor_smiles = [smiles]
        elif inter_size == 2:
            anchor_smiles = [get_anchor_smiles_metal(emol, a, metal_points, idxfunc) for a in anchors] #^ added new parameter metal_points
        else:
            anchors = [a for a in attach_points if is_anchor(emol.GetAtomWithIdx(a), [0])] #all attach points are labeled with 1
            attach_points = [a for a in attach_points if a not in anchors]
            attach_points = [anchors[0]] + attach_points + [anchors[1]] #force the attach_points to be a chain like anchor ... anchor
            anchor_smiles = [get_anchor_smiles_metal(emol, a, metal_points, idxfunc) for a in anchors] #^ added new parameter metal_points

        assert len(anchors) <= 2

        # ! One possible error with the below filtering of atoms in cluster can be that the continuous structure of the cluster can be broken if we remove the atoms which are in the highlight_atoms. So, i dont really know how it will affect the final output of attachment points in such cases. For now, can leave it like this and see how it pans out in larger dataset and future work. 

        cluster = [x for x in cluster if x not in highlight_atoms] # ^ added this to only consider atoms other than :2 in the parent cluster since we are currently only searching for cluster attachments needed for :1 atoms.

        if inter_size == 1:
            cands = [ [x] for x in cluster if x not in used ]

        elif anchor_smiles[0] == anchor_smiles[1]:
            cluster2 = cluster + cluster
            cands = [cluster2[i : i + inter_size] for i in range(len(cluster))] #not pairs if inter_size >= 3
            cands = [c for c in cands if (c[0], c[-1]) not in used
                    and bond_match(self.mol, c[0], c[-1], emol, attach_points[0], attach_points[-1]) ] #weak matching
        else: 
            cluster2 = cluster + cluster
            cands = [cluster2[i : i + inter_size] for i in range(len(cluster))]
            cluster2 = cluster2[::-1]
            cands += [cluster2[i : i + inter_size] for i in range(len(cluster))]
            cands = [c for c in cands if (c[0], c[-1]) not in used
                    and bond_match(self.mol, c[0], c[-1], emol, attach_points[0], attach_points[-1]) ] #weak matching

        if len(metal_points)>0:
            return cands, anchor_smiles, attach_points, False
        else:
            return cands, anchor_smiles, attach_points, True


