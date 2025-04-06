import torch
import rdkit
import numpy as np
import rdkit.Chem as Chem
import networkx as nx
from hgraph.chemutils import *
from hgraph.nnutils import *

add = lambda x,y : x + y if type(x) is int else (x[0] + y, x[1] + y)

class MolGraphMetal(object):

    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 70
    # MAX_POS = 100
    # MAX_POS=20

    def __init__(self, mol, highlight_atoms, xyz_block=None):
        # self.smiles = smiles
        # self.mol = get_mol(smiles)

        """
        Accessing the atom coordinates from the xyz block of the ligand and storing them as tuples.
        Initializing atom coordinates if the xyz block is provided. If not, then atom coordinates will be None.
        """
        atom_coordinates = []
        if xyz_block is not None:
            # print("XYZ block provided")
            data_lines=xyz_block.strip().split('\n')
            atom_data_lines = data_lines[2:]
            for i, line in enumerate(atom_data_lines):
                atom_info = line.split()
                _, x, y, z = atom_info
                x, y, z = float(x), float(y), float(z)
                atom_coordinates.append((x, y, z))
            
            self.atom_coordinates = atom_coordinates
            self.mol, self.highlight=self.renumber_molecule(mol, highlight_atoms)
            # print("length of highlight atoms: ", self.highlight)
            self.mol_graph=self.build_mol_graph_dist()
            self.clusters,self.atom_cls,self.clusters_xyz=self.find_clusters_dist()
            # print("Number of clusters: ", len(self.clusters))
            # print("atom_cls: ", self.atom_cls)
            # print("building tree")
            self.mol_tree, self.tree_motif_cords, self.flagged_motifs=self.tree_decomp_dist()
            # print("number of nodes: ", len(self.mol_tree.nodes))
            # print("number of edges: ", len(self.mol_tree.edges))
            self.order=self.label_tree_dist()
            # print("number of edges: ", len(self.mol_tree.edges))
            # print("order: ", self.order)
            # print("Number of nodes: ",  len(self.mol_tree.nodes))
            # print("Number of edges: ", len(self.mol_tree.edges))
            # print("Order:",self.order)
        
        else:
            print("No xyz block provided")
            self.atom_coordinates=None
            self.mol, self.highlight=self.renumber_molecule(mol,highlight_atoms)
            self.mol_graph = self.build_mol_graph()
            self.clusters, self.atom_cls = self.find_clusters()
            self.mol_tree, self.flagged_motifs = self.tree_decomp()
            self.order = self.label_tree()

        # if xyz_block is not None:
        #     self.atom_coordinates = atom_coordinates
        # else:
        #     self.atom_coordinates = None

        # self.mol = mol
        # self.highlight=highlight_atoms
        # self.mol_graph = self.build_mol_graph()
        # # self.clusters, self.atom_cls, self.clusters_xyz = self.find_clusters()
        # # self.mol_tree, self.tree_motif_cords = self.tree_decomp_dist()
        # self.clusters, self.atom_cls = self.find_clusters()
        # self.mol_tree = self.tree_decomp()
        # self.order = self.label_tree()

    def renumber_molecule(self, mol, highlight_atoms):
        num_atoms = mol.GetNumAtoms()
        all_atoms = list(range(num_atoms))
        
        non_highlight_atoms = [idx for idx in all_atoms if idx not in highlight_atoms]
        
        new_order = highlight_atoms + non_highlight_atoms
        new_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}
        
        new_highlight = [new_index_map[idx] for idx in highlight_atoms]
        
        mol = Chem.RenumberAtoms(mol, new_order)
        
        return mol, new_highlight

    def find_clusters(self):
        mol = self.mol
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1: #special case
            return [(0,)], [[0]]

        clusters = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                clusters.append( (a1,a2) )

        ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
        """
        ^ Temporary - Adding a new condition here while adding the rings to the clusters. We need to make sure that the rings detected are not just topological loops but actually chemically valid rings whose bonds are part of a ring. bond.IsInRing() fails because the molecule considers the topological loop as a ring itself. So, just putting a constraint on the size of the ring to be greater than 2 and lesser than 7 for now. Need to discuss this. 
        """
        for i, ring in enumerate(ssr):
            ring = list(ring)
            ring_size = len(ring)
            if ring_size > 2 and ring_size < 7:
                clusters.append(tuple(ring))

        if clusters and 0 not in clusters[0]: #root is not node[0]
            for i,cls in enumerate(clusters):
                if 0 in cls:
                    clusters = [clusters[i]] + clusters[:i] + clusters[i+1:]
                    #clusters[i], clusters[0] = clusters[0], clusters[i]
                    break

        atom_cls = [[] for i in range(n_atoms)]
        for i in range(len(clusters)):
            for atom in clusters[i]:
                atom_cls[atom].append(i)

        return clusters, atom_cls


    def find_clusters_dist(self):
        """
        clusters constructs bond and ring clusters and stores it. 

        atom_cls - stores the list of clusters each atom belongs to as a list of lists.

        clusters_xyz - stores the coordinats of the atoms in the clusters as a list of lists of coordinates of the atoms in each cluster.
        """
        clusters_xyz={} # added for distance calculation
        mol = self.mol
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1: #special case
            return [(0,)], [[0]]

        clusters = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                clusters.append( (a1,a2) )

        ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
        """
        ^ Temporary - Adding a new condition here while adding the rings to the clusters. We need to make sure that the rings detected are not just topological loops but actually chemically valid rings whose bonds are part of a ring. bond.IsInRing() fails because the molecule considers the topological loop as a ring itself. So, just putting a constraint on the size of the ring to be greater than 2 and lesser than 7 for now. Need to discuss this. 
        """
        for i, ring in enumerate(ssr):
            ring = list(ring)
            ring_size = len(ring)
            if ring_size > 2 and ring_size < 7:
                clusters.append(tuple(ring))

        if clusters and 0 not in clusters[0]: #root is not node[0]
            for i,cls in enumerate(clusters):
                if 0 in cls:
                    clusters = [clusters[i]] + clusters[:i] + clusters[i+1:]
                    #clusters[i], clusters[0] = clusters[0], clusters[i]
                    break

        atom_cls = [[] for i in range(n_atoms)]
        
        if self.atom_coordinates is None:
            for i in range(len(clusters)):
                for atom in clusters[i]:
                    atom_cls[atom].append(i)
            return clusters, atom_cls, None
        else:
            for i in range(len(clusters)):
                cluster_xyz=[]
                for atom in clusters[i]:
                    atom_cls[atom].append(i)
                    cluster_xyz.append(np.array(self.atom_coordinates[atom]))
                clusters_xyz[i]=cluster_xyz
            return clusters, atom_cls,clusters_xyz
        
    def tree_decomp(self):
        clusters_tree = self.clusters
        graph=nx.Graph()
        highlight_atoms=self.highlight
        flagged_motifs = []
        flagged_atoms = []

        for i in range(len(clusters_tree)):
            graph.add_node(i)

            count=0
            current_flag_atoms=[]
            for idx, atom in enumerate(clusters_tree[i]):
                if atom in highlight_atoms:
                    count+=1
                    current_flag_atoms.append(atom)
            if count>0 and len(clusters_tree[i])>2:
                flagged_motifs.append(i)
                flagged_atoms.extend(current_flag_atoms)
        
        for atom, nei_cls in enumerate(self.atom_cls):

            if len(nei_cls) <= 1: 
                if atom in highlight_atoms and atom not in flagged_atoms:
                    flagged_atoms.append(atom)
                    flagged_motifs.append(nei_cls[0])
                continue

            bonds = [c for c in nei_cls if len(clusters_tree[c]) == 2]
            rings = [c for c in nei_cls if len(clusters_tree[c]) > 4]

            if len(nei_cls) > 2 and len(bonds) >= 2:
                clusters_tree.append([atom])
                c2 = len(clusters_tree)-1
                graph.add_node(c2)
                if atom in highlight_atoms and atom not in flagged_atoms:
                    flagged_atoms.append(atom)
                    flagged_motifs.append(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)

            elif len(rings) > 2: #Bee Hives, len(nei_cls) > 2 
                clusters_tree.append([atom]) #temporary value, need to change
                c2 = len(clusters_tree)-1
                graph.add_node(c2)
                if atom in highlight_atoms and atom not in flagged_atoms:
                    flagged_atoms.append(atom)
                    flagged_motifs.append(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)
            else:
                for i,c1 in enumerate(nei_cls):
                    for c2 in nei_cls[i + 1:]:
                        inter = set(clusters_tree[c1]) & set(clusters_tree[c2])
                        graph.add_edge(c1, c2, weight = len(inter))

        # self.clusters=clusters_tree # commenting this because the original code does not update the clusters
        n, m = len(graph.nodes), len(graph.edges)
        if n-m==1:
            mst=graph
        else:
            mst=nx.maximum_spanning_tree(graph) #must be connected
        assert sorted(flagged_atoms) == sorted(highlight_atoms)
        return mst, flagged_motifs


    def tree_decomp_dist(self):
        clusters_tree = self.clusters
        print("clusters before tree decomposition")
        print(clusters_tree)
        highlight_atoms=self.highlight
        flagged_motifs = []
        flagged_atoms = []

        clusters_xyz=self.clusters_xyz
        tree_motif_cords = [None] * len(clusters_tree)

        graph=nx.Graph()
        for i in range(len(clusters_tree)):
            graph.add_node(i)

            """
            These are for the ring motifs. If the motif has more than 2 atoms and contains more than 1 highlighted atoms, then that is a ring motif with few of its atoms connected to the metal centre so then it is a flagged motif.
            """
            count=0
            current_flag_atoms=[]
            for idx, atom in enumerate(clusters_tree[i]):
                if atom in highlight_atoms:
                    count+=1
                    current_flag_atoms.append(atom)
            if count>0 and len(clusters_tree[i])>2:
                flagged_motifs.append(i)
                flagged_atoms.extend(current_flag_atoms)
        
        for atom, nei_cls in enumerate(self.atom_cls):
            if len(nei_cls) <= 1: 
                if atom in highlight_atoms and atom not in flagged_atoms:
                    flagged_atoms.append(atom)
                    flagged_motifs.append(nei_cls[0])
                
                if tree_motif_cords[nei_cls[0]] is None:
                    avg_c1 = np.mean(np.array(clusters_xyz[nei_cls[0]]), axis=0)
                    tree_motif_cords[nei_cls[0]]=avg_c1
                continue

            bonds = [c for c in nei_cls if len(clusters_tree[c]) == 2]
            rings = [c for c in nei_cls if len(clusters_tree[c]) > 4]

            if len(nei_cls) > 2 and len(bonds) >= 2:
                clusters_tree.append([atom])
                c2 = len(clusters_tree)-1
                tree_motif_cords.append(None)
                graph.add_node(c2)
                if atom in highlight_atoms and atom not in flagged_atoms:
                    flagged_atoms.append(atom)
                    flagged_motifs.append(c2)
                for c1 in nei_cls:
                    if clusters_xyz is not None:
                        avg_c1 = np.mean(np.array(clusters_xyz[c1]), axis=0) 
                        coord_c2=np.array(self.atom_coordinates[atom])

                        if tree_motif_cords[c1] is None:
                            tree_motif_cords[c1]=avg_c1
                        if tree_motif_cords[c2] is None:
                            tree_motif_cords[c2]=coord_c2

                        graph.add_edge(c1, c2, weight = 100, euclidean_distance = np.linalg.norm(avg_c1 - coord_c2))
                    else:
                        graph.add_edge(c1, c2, weight = 100)

            elif len(rings) > 2: #Bee Hives, len(nei_cls) > 2 
                clusters_tree.append([atom]) #temporary value, need to change
                c2 = len(clusters_tree)-1
                tree_motif_cords.append(None)
                graph.add_node(c2)
                if atom in highlight_atoms and atom not in flagged_atoms:
                    flagged_atoms.append(atom)
                    flagged_motifs.append(c2)
                for c1 in nei_cls:
                    if clusters_xyz is not None:
                        avg_c1 = np.mean(np.array(clusters_xyz[c1]), axis=0) 
                        coord_c2=np.array(self.atom_coordinates[atom])

                        if tree_motif_cords[c1] is None:
                            tree_motif_cords[c1]=avg_c1
                        if tree_motif_cords[c2] is None:
                            tree_motif_cords[c2]=coord_c2

                        graph.add_edge(c1, c2, weight = 100, euclidean_distance = np.linalg.norm(avg_c1 - coord_c2))
                    else:
                        graph.add_edge(c1, c2, weight = 100)

            else:
                for i,c1 in enumerate(nei_cls):
                    for c2 in nei_cls[i + 1:]:
                        inter = set(clusters_tree[c1]) & set(clusters_tree[c2])
                        """
                        Storing the euclidean distance between the clusters as an attribute of the edge in the graph. This is at the cluster level.
                        """
                        if clusters_xyz is not None:
                            avg_c1 = np.mean(np.array(clusters_xyz[c1]), axis=0) 
                            avg_c2 = np.mean(np.array(clusters_xyz[c2]), axis=0) 

                            if tree_motif_cords[c1] is None:
                                tree_motif_cords[c1]=avg_c1
                            if tree_motif_cords[c2] is None:
                                tree_motif_cords[c2]=avg_c2

                            graph.add_edge(c1, c2, weight=len(inter), euclidean_distance = np.linalg.norm(avg_c1 - avg_c2))
                        else:
                            graph.add_edge(c1, c2, weight = len(inter))

        # self.clusters=clusters_tree # commenting this because the original code does not update the clusters
        n, m = len(graph.nodes), len(graph.edges)
        if n-m==1:
            mst=graph
        else:
            mst=nx.maximum_spanning_tree(graph) #must be connected
        assert sorted(flagged_atoms) == sorted(highlight_atoms)
        return mst, tree_motif_cords, flagged_motifs

    def label_tree(self):
        def dfs(order, pa, prev_sib, x, fa):
            pa[x] = fa 
            #errorhandling
            if x in self.mol_tree:
                sorted_child = sorted([ y for y in self.mol_tree[x] if y != fa ]) #better performance with fixed order
            else:
                # raise Exception("Error in dfs of tree decomposition")
                print("Error in dfs of tree decomposition")
                return None
            for idx,y in enumerate(sorted_child):
                self.mol_tree[x][y]['label'] = 0 
                self.mol_tree[y][x]['label'] = idx + 1 #position encoding
                prev_sib[y] = sorted_child[:idx] 
                prev_sib[y] += [x, fa] if fa >= 0 else [x]
                order.append( (x,y,1) )
                dfs(order, pa, prev_sib, y, x)
                order.append( (y,x,0) )

        order, pa = [], {}
        self.mol_tree = nx.DiGraph(self.mol_tree)
        prev_sib = [[] for i in range(len(self.clusters))]
        dfs(order, pa, prev_sib, 0, -1)
        order.append( (0, None, 0) ) #last backtrack at root
        
        # TODO: Molecule recreation: new clean mol object might be necessary here (to ensure
        # fresh atom map numbers), so using self.mol might be wrong. Need to correct outer code accordingly. 
        # mol = get_mol(self.smiles) # modified this to remove smiles input dependency completely
        """
        Using Chem.RWMol(self.mol) instead of self.mol to create a fresh mol object for further processing as done by the authors in their original code with smiles strings. Since we have removed smiles dependency, we need to create a fresh mol object using Chem.RWMol() for further processing.
        """
        mol = Chem.RWMol(self.mol) #added
        for a in mol.GetAtoms():
            a.SetAtomMapNum( a.GetIdx() + 1 )

        tree = self.mol_tree
        highlights=self.highlight
        for i,cls in enumerate(self.clusters):
            inter_atoms = set(cls) & set(self.clusters[pa[i]]) if pa[i] >= 0 else set([0])
            cmol, inter_label = get_inter_label_metal(mol, cls, inter_atoms,highlights)
            # error handling temporary
            if cmol is None:
                return None
            tree.nodes[i]['ismiles'] = ismiles = get_smiles(cmol)
            tree.nodes[i]['inter_label'] = inter_label
            tree.nodes[i]['smiles'] = smiles = get_smiles(set_atommap(cmol))
            tree.nodes[i]['label'] = (smiles, ismiles) if len(cls) > 1 else (smiles, smiles)
            tree.nodes[i]['cluster'] = cls 
            tree.nodes[i]['assm_cands'] = []

            if pa[i] >= 0 and len(self.clusters[ pa[i] ]) > 2: #uncertainty occurs in assembly
                hist = [a for c in prev_sib[i] for a in self.clusters[c]] 
                pa_cls = self.clusters[ pa[i] ]
                tree.nodes[i]['assm_cands'] = get_assm_cands(mol, hist, inter_label, pa_cls, len(inter_atoms)) 

                #debugging/error handling temporary
                if tree.nodes[i]['assm_cands'] is None:
                    return None


                child_order = tree[i][pa[i]]['label']
                diff = set(cls) - set(pa_cls)
                for fa_atom in inter_atoms:
                    for ch_atom in self.mol_graph[fa_atom]:
                        if ch_atom in diff:
                            label = self.mol_graph[ch_atom][fa_atom]['label']
                            if type(label) is int: #in case one bond is assigned multiple times
                                self.mol_graph[ch_atom][fa_atom]['label'] = (label, child_order)
        return order
    
    def label_tree_dist(self):
        def dfs(order, pa, prev_sib, x, fa):
            pa[x] = fa 
            #errorhandling
            if x in self.mol_tree:
                sorted_child = sorted([ y for y in self.mol_tree[x] if y != fa ]) #better performance with fixed order
            else:
                raise Exception("Error in dfs of tree decomposition")
                # return None
            for idx,y in enumerate(sorted_child):
                self.mol_tree[x][y]['label'] = 0 
                self.mol_tree[y][x]['label'] = idx + 1 #position encoding
                prev_sib[y] = sorted_child[:idx] 
                prev_sib[y] += [x, fa] if fa >= 0 else [x]
                order.append( (x,y,1) )
                dfs(order, pa, prev_sib, y, x)
                order.append( (y,x,0) )

        order, pa = [], {}
        self.mol_tree = nx.DiGraph(self.mol_tree)
        prev_sib = [[] for i in range(len(self.clusters))]
        dfs(order, pa, prev_sib, 0, -1)
        order.append( (0, None, 0) ) #last backtrack at root
        
        # TODO: Molecule recreation: new clean mol object might be necessary here to ensure
        # fresh atom map numbers), so using self.mol might be wrong. Need to correct outer code accordingly. 
        # mol = get_mol(self.smiles) # modified this to remove smiles input dependency completely
        """
        Using Chem.RWMol(self.mol) instead of self.mol to create a fresh mol object for further processing as done by the authors in their original code with smiles strings. Since we have removed smiles dependency, we need to create a fresh mol object using Chem.RWMol() for further processing.
        """
        mol = Chem.RWMol(self.mol) #added
        for a in mol.GetAtoms():
            a.SetAtomMapNum( a.GetIdx() + 1 )

        tree = self.mol_tree
        highlights=self.highlight
        for i,cls in enumerate(self.clusters):
            inter_atoms = set(cls) & set(self.clusters[pa[i]]) if pa[i] >= 0 else set([0])
            cmol, inter_label = get_inter_label_metal(mol, cls, inter_atoms,highlights)
            if cmol is None:
                continue
            tree.nodes[i]['ismiles'] = ismiles = get_smiles(cmol)
            tree.nodes[i]['inter_label'] = inter_label
            tree.nodes[i]['smiles'] = smiles = get_smiles(set_atommap(cmol))
            tree.nodes[i]['label'] = (smiles, ismiles) if len(cls) > 1 else (smiles, smiles)
            tree.nodes[i]['cluster'] = cls 
            tree.nodes[i]['assm_cands'] = []
            tree.nodes[i]['coordinates'] = self.tree_motif_cords[i]

            """
            There is some sort of uncertainity in the assembly when the parent cluster is a ring and has more than 2 atoms. 
            """
            if pa[i] >= 0 and len(self.clusters[ pa[i] ]) > 2: #uncertainty occurs in assembly
                hist = [a for c in prev_sib[i] for a in self.clusters[c]] 
                pa_cls = self.clusters[ pa[i] ]
                tree.nodes[i]['assm_cands'] = get_assm_cands(mol, hist, inter_label, pa_cls, len(inter_atoms)) 

                child_order = tree[i][pa[i]]['label']
                diff = set(cls) - set(pa_cls)
                for fa_atom in inter_atoms:
                    for ch_atom in self.mol_graph[fa_atom]:
                        if ch_atom in diff:
                            label = self.mol_graph[ch_atom][fa_atom]['label']
                            if type(label) is int: #in case one bond is assigned multiple times
                                self.mol_graph[ch_atom][fa_atom]['label'] = (label, child_order)
        return order
    
    """
    modifying build_mol_graph to include whether the atoms of the ligand are connected to the metal centre or not as a feature of each atom(node) in the graph.
    """
    def build_mol_graph(self):
        mol = self.mol
        highlight_atom=self.highlight

        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))

        if highlight_atom is not None:
            highlight_set=set(highlight_atom)
        else:
            highlight_set=set()

        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())
            graph.nodes[atom.GetIdx()]['highlight'] = 1 if atom.GetIdx() in highlight_set else 0

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = MolGraphMetal.BOND_LIST.index( bond.GetBondType() )
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype

        return graph
    
    def build_mol_graph_dist(self):
        mol = self.mol
        highlight_atom=self.highlight

        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))

        if highlight_atom is not None:
            highlight_set=set(highlight_atom)
        else:
            highlight_set=set()

        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())
            graph.nodes[atom.GetIdx()]['highlight'] = 1 if atom.GetIdx() in highlight_set else 0

            """
            Extraction of the coordinates of the atoms from self.atom_coordinates and storing them as an attribute of the node in the graph.
            """
            if self.atom_coordinates is not None:
                coordinates=self.atom_coordinates[atom.GetIdx()]
                graph.nodes[atom.GetIdx()]['coordinates'] = coordinates

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = MolGraphMetal.BOND_LIST.index( bond.GetBondType() )
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype

            """
            Extracting the coordinates of the atoms, calculating the distance between them and storing it as an attribute of the edge in the graph. This is at the atom level. 
            """
            if self.atom_coordinates is not None:
                coord1 = self.atom_coordinates[a1]
                coord2 = self.atom_coordinates[a2]
                # print(coord1)
                # print(np.array(coord1))
                distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
                graph[a1][a2]['euclidean_distance'] = distance
                graph[a2][a1]['euclidean_distance'] = distance

        return graph
    
    @staticmethod
    def tensorize_metal(complexes_names_batch, complexes_ligands, complexes_highlights, complexes_ligandblock, complexes_iron_coord, vocab, avocab):
        mol_batch_metal_map = {}
        mol_tree_metal_map = {}
        mol_graph_metal_map = {}

        for idx, complex in enumerate(complexes_names_batch):
            ligands_set = complexes_ligands[complex]
            ligands_highlights = complexes_highlights[complex]

            # error handling temporary 
            # for i,ligand in enumerate(ligands_set):
            #     hmol=MolGraphMetal(ligand,ligands_highlights[i+1])
            #     if hmol.order is None:
            #         print("Error in order of the ligand")
            #         continue

            #     if complex not in mol_batch_metal_map:
            #         mol_batch_metal_map[complex] = []
            #     mol_batch_metal_map[complex].append(hmol)
            mol_batch_metal_map[complex] = [MolGraphMetal(ligand,ligands_highlights[i+1]) for i,ligand in enumerate(ligands_set)]

            # error handling temporary
            # if complex not in mol_batch_metal_map:
            #     continue
            mol_tree_metal_map[complex] = [x.mol_tree for x in mol_batch_metal_map[complex]]
            mol_graph_metal_map[complex] = [x.mol_graph for x in mol_batch_metal_map[complex]]
        
        tree_tensors, tree_batchG = MolGraphMetal.tensorize_graph_metal(complexes_names_batch, mol_tree_metal_map, vocab)
        graph_tensors, graph_batchG = MolGraphMetal.tensorize_graph_metal(complexes_names_batch, mol_graph_metal_map, avocab,('Fe',0), use_highlights=True)

        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1] 

        # error handling temporary
        # max_cls_size = 0
        # key_count=0
        # for x in complexes_names_batch:
        #     if x not in mol_batch_metal_map:
        #         continue
        #     key_count+=1
        #     for mgm in mol_batch_metal_map[x]:
        #         for c in mgm.clusters:
        #             max_cls_size = max(max_cls_size, len(c))
        # cgraph = torch.zeros(len(tree_batchG) + 1 + key_count, max_cls_size).int()
        max_cls_size=max([len(c) for x in complexes_names_batch for mgm in mol_batch_metal_map[x] for c in mgm.clusters])
        cgraph = torch.zeros(len(tree_batchG) + 1 + len(complexes_names_batch), max_cls_size).int()

        for v,attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['inter_label'] = inter_label = [(x + offset, y) for x,y in attr['inter_label']]
            tree_batchG.nodes[v]['cluster'] = cls = [x + offset for x in attr['cluster']]
            tree_batchG.nodes[v]['assm_cands'] = [add(x, offset) for x in attr['assm_cands']]
            cgraph[v, :len(cls)] = torch.IntTensor(cls)
        
        all_orders = []
        i = 0
        for mol_list in mol_batch_metal_map.values():
            for hmol in mol_list:
                offset=tree_scope[i][0]
                order = [(x + offset, y + offset, z) for x,y,z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)]
                all_orders.append(order)
                i += 1

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)
        return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders
    
    def tensorize_graph_metal(complexes_names_batch, mol_metal_map_batch, vocab, iron_tuple=('[Fe]', '[Fe:2]'), use_highlights=False):
        fnode, fmess = [None], [(0, 0, 0, 0)]
        agraph, bgraph = [[]], [[]]
        scope = []
        edge_dict = {}
        all_G = []
        bid=0

        for i,complex_name in enumerate(complexes_names_batch):
            if complex_name not in mol_metal_map_batch:
                continue
            graphs = mol_metal_map_batch[complex_name]
            print(complex_name)
            print(i)

            iron_index = len(fnode)
            fnode.append(vocab[iron_tuple])
            agraph.append([])

            for G in graphs:
                offset = len(fnode)
                scope.append((offset, len(G)))
                G = nx.convert_node_labels_to_integers(G, first_label=offset)
                all_G.append(G)
                fnode.extend([None for v in G.nodes])

                for v, attr in G.nodes(data='label'):
                    G.nodes[v]['batch_id'] = bid
                    fnode[v] = vocab[attr]
                    agraph.append([])
                
                bid += 1

                """
                Storing the new bonds information between the new iron node and the ligands in the graph. At the atom level, it will be just the bond type. To take care of this, we are defining a new bond type for the iron-ligand bonds denoted by index 4. Previously other 4 bond types were defined as 0, 1, 2, 3 for single, double, triple, and aromatic bonds respectively.
                """
                if use_highlights:
                    for v, attr in G.nodes(data='highlight'):
                        if attr == 1:  # Check if the node is highlighted
                            fmess.append((iron_index, v, 4, 0)) # New bond type index 4 for iron-ligand bonds
                            edge_dict[(iron_index, v)] = eid = len(edge_dict) + 1
                            agraph[v].append(eid)
                            bgraph.append([])
                else:
                    for v, attr in G.nodes(data='label'):
                        if ':2' in attr[1]:  # Check if the ismiles label contains ':2'
                            fmess.append((iron_index, v, -1, -1))  # Initialize it with -1 for now because we will need sorted order for this later on.
                            edge_dict[(iron_index, v)] = eid = len(edge_dict) + 1
                            agraph[v].append(eid)
                            bgraph.append([])

                """
                For attributes of edges at the ATOM level, the type can be a tuple because some modification is being done in the assm_cands part of the code in label_tree function for the self.mol_graph object. As of now, i have only encountered the tuple type at the atom level and not at the motif level. For the motif level, the edge attribute is a positional encoding as stored during the dfs traversal of the tree.
                """
                for u, v, attr in G.edges(data='label'):
                    if type(attr) is tuple:
                        fmess.append((u, v, attr[0], attr[1]))
                    else:
                        fmess.append((u, v, attr, 0))
                    edge_dict[(u, v)] = eid = len(edge_dict) + 1
                    G[u][v]['mess_idx'] = eid
                    agraph[v].append(eid)
                    bgraph.append([])

                for u, v in G.edges:
                    eid = edge_dict[(u, v)]
                    for w in G.predecessors(u):
                        if w == v:
                            continue
                        bgraph[eid].append(edge_dict[(w, u)])
            
            """
            Adding the bond information between the iron node and the ligands in the graph at the motif level. We sort the nodes which are attached to the iron node and then assign the index as the positional encoding at the tree level for each motif. 
            """
            neg_nodes = [fm[1] for fm in fmess if fm[2] == -1 and fm[3] == -1]
            sorted_neg_nodes = sorted(neg_nodes)
            neg_node_map = {node: idx+1 for idx, node in enumerate(sorted_neg_nodes)}
            for i, fm in enumerate(fmess):
                if fm[2] == -1 and fm[3] == -1:
                    updated_fmess = (fm[0], fm[1], neg_node_map[fm[1]], 0)
                    fmess[i] = updated_fmess

            neg_nodes = [fm[1] for fm in fmess if fm[2] == -1 and fm[3] == -1]

        fnode[0] = fnode[1]  # Set the first node to the iron node
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)

        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)
    
    @staticmethod
    def tensorize_metal_dist(complexes_names_batch, complexes_ligands, complexes_highlights, complexes_ligandblock, complexes_iron_coord, vocab, avocab):
        """
            INPUTS:
            complexes_names_batch - list of complex names in the batch.
            complexes_ligands - dictionary with complex names as keys and the set of ligands in the complex as values.
            complexes_highlights - dictionary with complex names as keys and the highlights (atom indices correspondong to the atoms which are attached to metal_center) of the ligands in the complex as values.
            complexes_ligandblock - dictionary with complex names as keys and the x,y,z information of the ligands in the complex as values.
            complexes_iron_coord - dictionary with complex names as keys and the iron coordinates of the complexes as values.
            vocab - vocabulary for the ligands.
            avocab - vocabulary for the atoms.


            mol_batch_metal_map - stores the MolGraphMetal objects for each ligand in the complexes in the batch.
            mol_tree_metal_map - stores the mol_tree of the ligands in the complexes in the batch.
            mol_graph_metal_map - stores the mol_graph of the ligands in the complexes in the batch.
            mol_ironcoord_map - stores the iron coordinates of the complexes in the batch.

            tree_scope and graph_scope inside the tree_tensors and graph_tensors are the scope tensors storing the root nodes index alongwith the length of that corresponding complex for the tree and graph tensors respectively.

            treescope_allorders and graphscope_allorders store the offset for each ligand within each complex to be used in the offset for cgraph. 

            all_orders - stores the dfs ordering fore ach complex in the batch set by the offset.

        """
        mol_batch_metal_map = {}
        mol_tree_metal_map = {}
        mol_graph_metal_map = {}
        mol_flagged_motifs_metal_map = {}
        mol_ironcoord_map = {}
        print(complexes_names_batch)
        # print(len(complexes_ligands))

        complex_ligand_count = []

        for idx, complex in enumerate(complexes_names_batch):
            print(complex)
            ligands_set = complexes_ligands[complex]
            print(f"number of ligands in complex: ",len(ligands_set))
            complex_ligand_count.append(len(ligands_set))
            ligands_highlights = complexes_highlights[complex]
            ligands_block = complexes_ligandblock[complex]
            iron_coord=complexes_iron_coord[complex] # modification for distance calculation
            mol_ironcoord_map[complex] = iron_coord
        
            mol_batch_metal_map[complex]=[MolGraphMetal(ligand,ligands_highlights[i+1],ligands_block[i]) for i,ligand in enumerate(ligands_set)]
            # for i,x in enumerate(mol_batch_metal_map[complex]):
            #     print(f"ligand {i} length: ",len(x.mol_graph.nodes))
            mol_tree_metal_map[complex] = [x.mol_tree for x in mol_batch_metal_map[complex]]
            mol_graph_metal_map[complex] = [x.mol_graph for x in mol_batch_metal_map[complex]]
            mol_flagged_motifs_metal_map[complex]=[x.flagged_motifs for x in mol_batch_metal_map[complex]]
        
        print("tree tensors")
        tree_tensors,tree_batchG,treescope_allorders,all_orders =MolGraphMetal.tensorize_graph_metal_dist(complexes_names_batch, mol_flagged_motifs_metal_map, mol_tree_metal_map, vocab, mol_ironcoord_map=mol_ironcoord_map)
        print("Number of nodes in the tree ",len(tree_batchG))
        print("graph tensors")
        graph_tensors, graph_batchG,graphscope_allorders = MolGraphMetal.tensorize_graph_metal_dist(complexes_names_batch, mol_flagged_motifs_metal_map, mol_graph_metal_map, avocab, mol_ironcoord_map,('Fe',0), use_highlights=True)
        print("Number of nodes in the graph ",len(graph_batchG))

        print("length of all orders: ",len(all_orders))
        print("all_orders for the first complex: ",all_orders[0])
        # exit()


        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1] 

        max_cls_size=max([len(c) for x in complexes_names_batch for mgm in mol_batch_metal_map[x] for c in mgm.clusters])
        cgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).int()

        """
        Corresponding to each node in the tree (i.e. motif), we store the corresponding atoms of that motif in the graph as a cluster in cgraph. So cgraph is a tensor of size [(number of motifs, max number of atoms in a motif)]. each cgraph entry consists of the atom indices of that cluster. 
        """
        print("cgraph")
        root_scope_iter=0
        for v,attr in tree_batchG.nodes(data=True):
            if attr['batch_id'] is not None:
                bid = attr['batch_id']
                offset = graphscope_allorders[bid][0]
                # print("offset: ",offset)
                tree_batchG.nodes[v]['inter_label'] = inter_label = [(x + offset, y) for x,y in attr['inter_label']]
                tree_batchG.nodes[v]['cluster'] = cls = [x +offset for x in attr['cluster']]
                tree_batchG.nodes[v]['assm_cands'] = [add(x, offset) for x in attr['assm_cands']]
                cgraph[v, :len(cls)] = torch.IntTensor(cls)
                # print(cls)
            else:
                offset = graph_scope[root_scope_iter][0]
                # print("offset root: ",offset)
                tree_batchG.nodes[v]['cluster']=cls=[x + offset for x in attr['cluster']]
                tree_batchG.nodes[v]['assm_cands']=[add(x, offset) for x in attr['assm_cands']]
                cgraph[v,:len(cls)]=torch.IntTensor(cls)
                root_scope_iter+=1
                # print(cls)
        
        # all_orders = []
        # i = 0
        # for mol_list in mol_batch_metal_map.values():
        #     for hmol in mol_list:
        #         offset=tree_scope[i][0]
        #         order = [(x + offset, y + offset, z) for x,y,z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)]
        #         all_orders.append(order)
        #         i += 1

        # all_orders = []
        # i=0
        # for mol_list in mol_batch_metal_map.values():
        #     cumulative_order = []  # To accumulate tuples for all hmol in the current mol_list
        #     for hmol in mol_list:
        #         offset = treescope_allorders[i][0]
        #         cumulative_order.extend([(x + offset, y + offset, z) for x, y, z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)])
        #         i += 1
        #     all_orders.append(cumulative_order)  # Append accumulated order for the entire mol_list

        print(all_orders)

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)

        return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders, complex_ligand_count

    # def tensorize_metal_dist(complexes_names_batch, complexes_ligands, complexes_highlights, complexes_ligandblock, complexes_iron_coord, vocab, avocab):
    #     """
    #     INPUTS:
    #     complexes_names_batch - list of complex names in the batch.
    #     complexes_ligands - dictionary with complex names as keys and the set of ligands in the complex as values.
    #     complexes_highlights - dictionary with complex names as keys and the highlights (atom indices correspondong to the atoms which are attached to metal_center) of the ligands in the complex as values.
    #     complexes_ligandblock - dictionary with complex names as keys and the x,y,z information of the ligands in the complex as values.
    #     complexes_iron_coord - dictionary with complex names as keys and the iron coordinates of the complexes as values.
    #     vocab - vocabulary for the ligands.
    #     avocab - vocabulary for the atoms.


    #     mol_batch_metal_map - stores the MolGraphMetal objects for each ligand in the complexes in the batch.
    #     mol_tree_metal_map - stores the mol_tree of the ligands in the complexes in the batch.
    #     mol_graph_metal_map - stores the mol_graph of the ligands in the complexes in the batch.
    #     mol_ironcoord_map - stores the iron coordinates of the complexes in the batch.

    #     """
    #     mol_batch_metal_map = {}
    #     mol_tree_metal_map = {}
    #     mol_graph_metal_map = {}
    #     mol_ironcoord_map = {}
    #     print(complexes_names_batch)
    #     # print(len(complexes_ligands))

    #     for idx, complex in enumerate(complexes_names_batch):
    #         print(complex)
    #         ligands_set = complexes_ligands[complex]
    #         print(len(ligands_set))
    #         ligands_highlights = complexes_highlights[complex]
    #         ligands_block = complexes_ligandblock[complex]
    #         iron_coord=complexes_iron_coord[complex] # modification for distance calculation
    #         mol_ironcoord_map[complex] = iron_coord
        
    #         mol_batch_metal_map[complex]=[MolGraphMetal(ligand,ligands_highlights[i+1],ligands_block[i]) for i,ligand in enumerate(ligands_set)]
    #         for i,x in enumerate(mol_batch_metal_map[complex]):
    #             print(f"ligand {i} length: ",len(x.mol_graph.nodes))
    #         mol_tree_metal_map[complex] = [x.mol_tree for x in mol_batch_metal_map[complex]]
    #         mol_graph_metal_map[complex] = [x.mol_graph for x in mol_batch_metal_map[complex]]

    #     print("tree tensors")
    #     tree_tensors,tree_batchG,treescope_allorders=MolGraphMetal.tensorize_graph_metal_dist(complexes_names_batch, mol_tree_metal_map, vocab, mol_ironcoord_map=mol_ironcoord_map)
    #     print("graph tensors")
    #     graph_tensors, graph_batchG,graphscope_allorders = MolGraphMetal.tensorize_graph_metal_dist(complexes_names_batch, mol_graph_metal_map, avocab, mol_ironcoord_map,('Fe',0), use_highlights=True)

    #     tree_scope = tree_tensors[-1]
    #     graph_scope = graph_tensors[-1] 

    #     max_cls_size=max([len(c) for x in complexes_names_batch for mgm in mol_batch_metal_map[x] for c in mgm.clusters])
    #     cgraph = torch.zeros(len(tree_batchG) + 1 + len(complexes_names_batch), max_cls_size).int()

    #     """
    #     Corresponding to each node in the tree (i.e. motif), we store the corresponding atoms of that motif in the graph as a cluster in cgraph. So cgraph is a tensor of size (number of motifs, max number of atoms in a motif). each cgraph entry consists of the atom indices of that cluster. 
    #     """
    #     # print("cgraph")
    #     for v,attr in tree_batchG.nodes(data=True):
    #         bid = attr['batch_id']
    #         offset = graphscope_allorders[bid][0]
    #         tree_batchG.nodes[v]['inter_label'] = inter_label = [(x + offset, y) for x,y in attr['inter_label']]
    #         tree_batchG.nodes[v]['cluster'] = cls = [x + offset for x in attr['cluster']]
    #         tree_batchG.nodes[v]['assm_cands'] = [add(x, offset) for x in attr['assm_cands']]
    #         cgraph[v, :len(cls)] = torch.IntTensor(cls)
        
    #     # all_orders = []
    #     # i = 0
    #     # for mol_list in mol_batch_metal_map.values():
    #     #     for hmol in mol_list:
    #     #         offset=tree_scope[i][0]
    #     #         order = [(x + offset, y + offset, z) for x,y,z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)]
    #     #         all_orders.append(order)
    #     #         i += 1

    #     all_orders = []
    #     i=0
    #     for mol_list in mol_batch_metal_map.values():
    #         cumulative_order = []  # To accumulate tuples for all hmol in the current mol_list
    #         for hmol in mol_list:
    #             offset = treescope_allorders[i][0]
    #             cumulative_order.extend([(x + offset, y + offset, z) for x, y, z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)])
    #             i += 1
    #         all_orders.append(cumulative_order)  # Append accumulated order for the entire mol_list


    #     tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)
    #     return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders
    
    def tensorize_graph_metal_dist(complexes_names_batch, mol_flagged_motifs_metal_map, mol_metal_map_batch, vocab, mol_ironcoord_map, iron_tuple=('[Fe]', '[Fe:2]'), use_highlights=False):
        """
        fnode - unique node indices correspondning to the vocabulary of the motifs and atoms in the graph.

        fmess - edge information between the nodes in the graph. It consists of the source node, target node, bond_type(for atoms)/position_encoding(for motifs), child_order(for atoms)/0(for motifs), and the euclidean distance.

        agraph - stores all the incoming edge ids for each node in the graph.

        bgraph - for a node v, let u be the node such that there exists u->v. Then bgraph stores the edge ids of the incoming edges to u. The indexing of bgraph is done based on the edge id of the edge from u->v.

        scope - stores the offset and length of each complex in the batch. Basically signifies the root node of each complex in the batch and the length of the complex from that root node.

        scope_allorders - stores the offset and length of each ligand in the batch. Basically signifies the root node of each ligand in the batch and the length of the ligand from that root node.

        edge_dict - stores the edge id of the edge between two nodes in the graph.

        all_G - stores the union of graph objects of all the complexes in the batch.

        bid - batch id for each ligand in the batch.

        #^ for each complex:
            
            #& 1. append the iron node manually as the root node to the fnode list and create a corresponding entry for the node in agraph. 

            #& 2. add the offset for this complex into the scope list.

            #& 3. create a networkx complex_graph object for the complex and add the iron node to the complex_graph with the corresponding attributes.

            #& 4. for each ligand in the complex:

                a. add the offset of this ligand to the scope_allorders list.

                b. append the nodes of the ligand to the fnode list and create a corresponding entry for the nodes in agraph.

                c. add the nodes and edges of the ligand with their attributes to the complex_graph. 

                d. add the bond information between the iron node and the attachment points at the ligands in the graph into fmess and the edge_id into agraph of the incoming node. At the same time, add the reverse edge as well. Create bgraph entries for both edges. (This is done both at atom and motif level. For the positional encoding at motif level, we keep it at -1 while adding and address this later. For atom level, the child order is taken as zero because the condition for assm_cands in label_tree is not satisfied when we see the parent atom as the iron atom.)

                e. now iterate over the edges of the ligand itself add edge information to fmess and the edge_id to agraph of the incoming node. Create bgraph entries for the edges.

            #& 5. add the bgraph entries for the edges within each ligand as well as between the iron node and the ligands using complex_graph. We are doing this instead of the previous approach because we need to consider the edges between the iron node and the ligands as well and individual ligand graphs G within the loop do not have that information. complex_graph has this information only after processing all the ligands.

            #& 6. add the positional encoding between the iron nodes and the nodes attached to the iron node in the graph at the motif level. We sort the nodes which are attached to the iron node and then assign the index as the positional encoding at the tree level for each motif. (This same procedure is done while assigning the positional encoding to the edges in the dfs traversal of the tree inside the label_tree function.)

            #& 7. for the scope of this complex, update the length of the complex with complex_length now that all the ligands are processed and we have the total length of the molecule. 

            #& 8. append the complex_graph object to the all_G list.

            #! One thing is missing is a motif with the iron node and its attached atom node? How else will the decoder know that it is a viable attachment point? or is it possible for the decoder to figure it out on itself. We are slightly changing our approach to align it with theirs now. We have modeled a complex as a full molecule at the tensor level. But there are a few gaps such as loops and the motif with the iron node and the attached atom node. We need to see how it goes.
        

        """

        fnode, fmess = [None], [(0, 0, 0, 0, 0)]
        agraph, bgraph = [[]], [[]]
        scope = []
        scope_allorders=[]
        edge_dict = {}
        all_G = []
        bid = 0
        trees_allorders=[]

        for complex_id, complex_name in enumerate(complexes_names_batch):
            print(complex_name)
            iron_coord=mol_ironcoord_map[complex_name] # modification for distance calculation
            graphs = mol_metal_map_batch[complex_name] # graphs are the ligands of this particular complex. 
            complex_flagged_motifs = mol_flagged_motifs_metal_map[complex_name]


            offset = len(fnode)
            scope.append((offset, len(graphs)))
            # scope_allorders.append((offset, 1))
            iron_index = len(fnode)
            # print("iron_index: ",iron_index)
            fnode.append(vocab[iron_tuple])
            agraph.append([])
            complex_length=1

            complex_graph=nx.DiGraph()#!
            if use_highlights:
                complex_graph.add_node(iron_index, coordinates=iron_coord, label=('Fe',0), highlight=None, batch_id=None)#!
            else:
                complex_graph.add_node(iron_index, ismiles='[Fe:2]', smiles='[Fe]', coordinates=iron_coord, label=iron_tuple, assm_cands=[], batch_id=None, inter_label=None,
                                       cluster=[0])#!

            for G,flagged_motifs in zip(graphs,complex_flagged_motifs):
                print("ligand length : ",len(G))
                complex_length+=len(G)
                offset = len(fnode)
                scope_allorders.append((offset, len(G)))
                G = nx.convert_node_labels_to_integers(G, first_label=offset)
                flagged_motifs = [v + offset for v in flagged_motifs]

                # all_G.append(G) #!
                fnode.extend([None for v in G.nodes])

                # print("motif indices")
                for v, attr in G.nodes(data='label'):
                    # print(v)
                    G.nodes[v]['batch_id'] = bid
                    fnode[v] = vocab[attr]
                    agraph.append([])
                
                bid += 1
                complex_graph = nx.compose(complex_graph, G) #!

                """
                Storing the new bonds information between the new iron node and the ligands in the graph. At the atom level, it will be just the bond type. To take care of this, we are defining a new bond type for the iron-ligand bonds denoted by index 4. Previously other 4 bond types were defined as 0, 1, 2, 3 for single, double, triple, and aromatic bonds respectively.

                use_highlights is done for graph level. else is done for motif level.
                """
                if use_highlights:

                    for v, attr in G.nodes(data=True):
                        if attr.get('highlight') == 1:  # Check if the node is highlighted
                            
                            coords=attr['coordinates']
                            dist=np.linalg.norm(np.array(iron_coord)-np.array(coords))

                            fmess.append((iron_index, v, 4, 0, dist)) # ! New bond type index 4 for iron-ligand bonds - assigning 0 for child order because the condition for assm_cands in label_tree is not satisfied when we see the parent atom as the iron atom.
                            edge_dict[(iron_index, v)] = eid = len(edge_dict) + 1
                            complex_graph.add_edge(iron_index, v, weight=1, label=4, euclidean_distance=dist)#!
                            complex_graph[iron_index][v]['mess_idx'] = eid
                            agraph[v].append(eid)
                            bgraph.append([])

                            # ! we need bidirectional edges so adding the reverse edge as well.
                            fmess.append((v, iron_index, 4, 0, dist))
                            edge_dict[(v, iron_index)] = eid = len(edge_dict) + 1
                            complex_graph.add_edge(v, iron_index, weight=1, label=4, euclidean_distance=dist)#!
                            complex_graph[v][iron_index]['mess_idx'] = eid
                            agraph[iron_index].append(eid)
                            bgraph.append([])
                            # break #! we are breaking to avoid formation of loops as seen in the example of FRMFTE10 ring.

                else:

                    for v, attr in G.nodes(data=True):
                        # label_attr=attr['label']
                        # if ':2' in label_attr[1]:  # Check if the ismiles label contains ':2'
                        if v in flagged_motifs:
                            coords=attr['coordinates']
                            dist=np.linalg.norm(np.array(iron_coord)-np.array(coords))

                            fmess.append((iron_index, v, 0, 0, dist))  # in the dfs ordering parent->child position encoding is put as zero so following the same here.
                            edge_dict[(iron_index, v)] = eid = len(edge_dict) + 1
                            complex_graph.add_edge(iron_index, v, weight=1, euclidean_distsance=dist, label=0)#!
                            complex_graph[iron_index][v]['mess_idx'] = eid
                            agraph[v].append(eid)
                            bgraph.append([])

                            # ! we need bidirectional edges so adding the reverse edge as well.
                            fmess.append((v, iron_index, -1, -1, dist)) # Initialize it with -1 for now because we will need sorted order for this later on to assign the positional encoding.
                            edge_dict[(v, iron_index)] = eid = len(edge_dict) + 1
                            complex_graph.add_edge(v, iron_index, weight=1, euclidean_distsance=dist, label=-1)#!
                            complex_graph[v][iron_index]['mess_idx'] = eid
                            agraph[iron_index].append(eid)
                            bgraph.append([])
                            # break #! we are breaking to avoid formation of loops as seen in the example of FRMFTE10 ring. 
                            
                            
                """
                For attributes of edges at the ATOM level, the type can be a tuple because some modification is being done in the assm_cands part of the code in label_tree function for the self.mol_graph object. As of now, i have only encountered the tuple type at the atom level and not at the motif level. For the motif level, the edge attribute is a single positional encoding as stored during the dfs traversal of the tree.
                """
                for u, v, attr in G.edges(data=True):
                    label_attr=attr['label']
                    distance=attr['euclidean_distance']
                    if type(label_attr) is tuple:
                        fmess.append((u, v, label_attr[0], label_attr[1], distance))
                    else:
                        fmess.append((u, v, label_attr, 0, distance))
                    edge_dict[(u, v)] = eid = len(edge_dict) + 1
                    # G[u][v]['mess_idx'] = eid
                    complex_graph[u][v]['mess_idx'] = eid
                    agraph[v].append(eid)
                    bgraph.append([])

                # for u, v in G.edges:
                #     eid = edge_dict[(u, v)]
                #     for w in G.predecessors(u):
                #         if w == v:
                #             continue
                #         bgraph[eid].append(edge_dict[(w, u)])
            
            #! Populating bgraph for the edges within each ligand as well as between the iron node and the ligands using complex_graph.
            for u,v in complex_graph.edges:
                eid=edge_dict[(u,v)]
                for w in complex_graph.predecessors(u):
                    if w==v:
                        continue
                    bgraph[eid].append(edge_dict[(w,u)])
            
            scope[complex_id]=(scope[complex_id][0],complex_length)
            """
            Adding the bond information between the iron node and the ligands in the graph at the motif level. We sort the nodes which are attached to the iron node and then assign the index as the positional encoding at the tree level for each motif. 
            """
            neg_nodes = [fm[0] for fm in fmess if fm[2] == -1 and fm[3] == -1]
            sorted_neg_nodes = sorted(neg_nodes)
            neg_node_map = {node: idx+1 for idx, node in enumerate(sorted_neg_nodes)}
            for i, fm in enumerate(fmess):
                if fm[2] == -1 and fm[3] == -1:
                    updated_fmess = (fm[0], fm[1], neg_node_map[fm[0]], 0, fm[4])
                    fmess[i] = updated_fmess

            all_G.append(complex_graph) #!
            if use_highlights==False:
                def dfs_complex(order,pa,prev_sib,x,fa,visited):
                    visited.add(x)             
                    pa[x]=fa
                    if x in complex_graph:
                        sorted_child = sorted([ y for y in complex_graph[x] if y != fa and y not in visited])
                    else:
                        raise Exception("Node not found in complex graph")
                    
                    for idx,y in enumerate(sorted_child):
                        complex_graph[x][y]['label'] = 0
                        complex_graph[y][x]['label'] = idx+1
                        prev_sib[y] = sorted_child[:idx] 
                        prev_sib[y] += [x, fa] if fa >= 0 else [x]
                        order.append( (x,y,1) )
                        dfs_complex(order, pa, prev_sib, y, x,visited)
                        order.append( (y,x,0) )
                order,pa=[],{}
                visited=set()
                # print(len(complex_graph))
                prev_sib=[[] for i in range(len(complex_graph)+iron_index)]
                dfs_complex(order,pa,prev_sib,iron_index,-1,visited)
                order.append((iron_index,None,0))
                trees_allorders.append(order)
                # print(order)

        fnode[0] = fnode[1]  # Set the first node to the iron node
        fnode = torch.IntTensor(fnode)
        fmess=torch.FloatTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)
        if use_highlights:
            return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G), scope_allorders
        else:
            return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G), scope_allorders, trees_allorders
    
    
    # def tensorize_graph_metal_dist(complexes_names_batch, mol_metal_map_batch, vocab, mol_ironcoord_map, iron_tuple=('Fe', 'Fe:2'), use_highlights=False):
    #     """
    #     fnode - unique node indices correspondning to the vocabulary of the motifs and atoms in the graph.

    #     fmess - edge information between the nodes in the graph. It consists of the source node, target node, bond_type(for atoms)/position_encoding(for motifs), child_order(for atoms)/0(for motifs), and the euclidean distance.

    #     agraph - stores all the incoming edge ids for each node in the graph.

    #     bgraph - for a node v, let u be the node such that there exists u->v. Then bgraph stores the edge ids of the incoming edges to u. The indexing of bgraph is done based on the edge id of the edge from u->v.
        

    #     """

    #     fnode, fmess = [None], [(0, 0, 0, 0, 0)]
    #     agraph, bgraph = [[]], [[]]
    #     scope = []
    #     scope_allorders=[]
    #     edge_dict = {}
    #     all_G = []
    #     bid = 0

    #     for complex_id, complex_name in enumerate(complexes_names_batch):
    #         iron_coord=mol_ironcoord_map[complex_name] # modification for distance calculation
    #         graphs = mol_metal_map_batch[complex_name]

    #         offset = len(fnode)
    #         scope.append((offset, len(graphs)))
    #         # scope_allorders.append((offset, 1))
    #         iron_index = len(fnode)
    #         fnode.append(vocab[iron_tuple])
    #         agraph.append([])
    #         complex_length=1

    #         for G in graphs:
    #             complex_length+=len(G)
    #             offset = len(fnode)
    #             scope_allorders.append((offset, len(G)))
    #             G = nx.convert_node_labels_to_integers(G, first_label=offset)
    #             all_G.append(G)
    #             fnode.extend([None for v in G.nodes])

    #             for v, attr in G.nodes(data='label'):
    #                 G.nodes[v]['batch_id'] = bid
    #                 fnode[v] = vocab[attr]
    #                 agraph.append([])
                
    #             bid += 1

    #             """
    #             Storing the new bonds information between the new iron node and the ligands in the graph. At the atom level, it will be just the bond type. To take care of this, we are defining a new bond type for the iron-ligand bonds denoted by index 4. Previously other 4 bond types were defined as 0, 1, 2, 3 for single, double, triple, and aromatic bonds respectively.

    #             use_highlights is done for graph level. else is done for motif level.
    #             """
    #             if use_highlights:
    #                 for v, attr in G.nodes(data=True):
    #                     if attr.get('highlight') == 1:  # Check if the node is highlighted
    #                         coords=attr['coordinates']
    #                         dist=np.linalg.norm(np.array(iron_coord)-np.array(coords))
    #                         fmess.append((iron_index, v, 4, 0, dist)) # ! New bond type index 4 for iron-ligand bonds - 0 is given for child order but need to put the logic of assm_cands here.
    #                         edge_dict[(iron_index, v)] = eid = len(edge_dict) + 1
    #                         agraph[v].append(eid)
    #                         bgraph.append([])
    #                         # ! we need bidirectional edges so adding the reverse edge as well.
    #                         fmess.append((v, iron_index, 4, 0, dist))
    #                         edge_dict[(v, iron_index)] = eid = len(edge_dict) + 1
    #                         agraph[iron_index].append(eid)
    #                         bgraph.append([])
    #             else:
    #                 for v, attr in G.nodes(data=True):
    #                     label_attr=attr['label']
    #                     if ':2' in label_attr[1]:  # Check if the ismiles label contains ':2'
    #                         coords=attr['coordinates']
    #                         dist=np.linalg.norm(np.array(iron_coord)-np.array(coords))
    #                         fmess.append((iron_index, v, -1, -1, dist))  # Initialize it with -1 for now because we will need sorted order for this later on.
    #                         edge_dict[(iron_index, v)] = eid = len(edge_dict) + 1
    #                         agraph[v].append(eid)
    #                         bgraph.append([])
    #                         # ! we need bidirectional edges so adding the reverse edge as well.
    #                         fmess.append((v, iron_index, -1, -1, dist))
    #                         edge_dict[(v, iron_index)] = eid = len(edge_dict) + 1
    #                         agraph[iron_index].append(eid)
    #                         bgraph.append([])

    #             """
    #             For attributes of edges at the ATOM level, the type can be a tuple because some modification is being done in the assm_cands part of the code in label_tree function for the self.mol_graph object. As of now, i have only encountered the tuple type at the atom level and not at the motif level. For the motif level, the edge attribute is a positional encoding as stored during the dfs traversal of the tree.
    #             """
    #             for u, v, attr in G.edges(data=True):
    #                 label_attr=attr['label']
    #                 distance=attr['euclidean_distance']
    #                 if type(label_attr) is tuple:
    #                     fmess.append((u, v, label_attr[0], label_attr[1], distance))
    #                 else:
    #                     fmess.append((u, v, label_attr, 0, distance))
    #                 edge_dict[(u, v)] = eid = len(edge_dict) + 1
    #                 G[u][v]['mess_idx'] = eid
    #                 agraph[v].append(eid)
    #                 bgraph.append([])

    #             for u, v in G.edges:
    #                 eid = edge_dict[(u, v)]
    #                 for w in G.predecessors(u):
    #                     if w == v:
    #                         continue
    #                     bgraph[eid].append(edge_dict[(w, u)])
            
    #         scope[complex_id]=(scope[complex_id][0],complex_length)
            
    #         """
    #         Adding the bond information between the iron node and the ligands in the graph at the motif level. We sort the nodes which are attached to the iron node and then assign the index as the positional encoding at the tree level for each motif. 
    #         """
    #         neg_nodes = [fm[1] for fm in fmess if fm[2] == -1 and fm[3] == -1]
    #         sorted_neg_nodes = sorted(neg_nodes)
    #         neg_node_map = {node: idx+1 for idx, node in enumerate(sorted_neg_nodes)}
    #         for i, fm in enumerate(fmess):
    #             if fm[2] == -1 and fm[3] == -1:
    #                 updated_fmess = (fm[0], fm[1], neg_node_map[fm[1]], 0, fm[4])
    #                 fmess[i] = updated_fmess

    #         neg_nodes = [fm[1] for fm in fmess if fm[2] == -1 and fm[3] == -1]
        

    #     fnode[0] = fnode[1]  # Set the first node to the iron node
    #     fnode = torch.IntTensor(fnode)
    #     fmess=torch.FloatTensor(fmess)
    #     # print(fmess.shape)
    #     # print(fmess.dtype)
    #     # print(fmess[6])
    #     # print(fmess[5,1])
    #     # print(fmess[5,2])
    #     # print(fmess[5,3])
    #     # print(fmess[5,4])
    #     # fmess = torch.IntTensor(fmess)
    #     # print(fmess[:,0].dtype)
    #     # print(fmess[:,1].dtype)
    #     # print(fmess[:,2].dtype)
    #     # print(fmess[:,3].dtype)
    #     # print(fmess[:,4].dtype)
    #     agraph = create_pad_tensor(agraph)
    #     bgraph = create_pad_tensor(bgraph)

    #     return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G), scope_allorders
        
if __name__ == "__main__":
    import sys
    
    test_smiles = ['CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1','O=C1OCCC1Sc1nnc(-c2c[nH]c3ccccc23)n1C1CC1', 'CCN(C)S(=O)(=O)N1CCC(Nc2cccc(OC)c2)CC1', 'CC(=O)Nc1cccc(NC(C)c2ccccn2)c1', 'Cc1cc(-c2nc3sc(C4CC4)nn3c2C#N)ccc1Cl', 'CCOCCCNC(=O)c1cc(OC)ccc1Br', 'Cc1nc(-c2ccncc2)[nH]c(=O)c1CC(=O)NC1CCCC1', 'C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F', 'CCOc1ccc(CN2c3ccccc3NCC2C)cc1N', 'NC(=O)C1CCC(CNc2cc(-c3ccccc3)nc3ccnn23)CC1', 'CC1CCc2noc(NC(=O)c3cc(=O)c4ccccc4o3)c2C1', 'c1cc(-n2cnnc2)cc(-n2cnc3ccccc32)c1', 'Cc1ccc(-n2nc(C)cc2NC(=O)C2CC3C=CC2C3)nn1', 'O=c1ccc(c[nH]1)C1NCCc2ccc3OCCOc3c12']

    for s in sys.stdin:#test_smiles:
        print(s.strip("\r\n "))
        #mol = Chem.MolFromSmiles(s)
        #for a in mol.GetAtoms():
        #    a.SetAtomMapNum( a.GetIdx() )
        #print(Chem.MolToSmiles(mol))

        hmol = MolGraphMetal(s)
        print(hmol.clusters)
        #print(list(hmol.mol_tree.edges))
        print(nx.get_node_attributes(hmol.mol_tree, 'label'))
        #print(nx.get_node_attributes(hmol.mol_tree, 'inter_label'))
        #print(nx.get_node_attributes(hmol.mol_tree, 'assm_cands'))
        #print(hmol.order)



# 1. the common atom vocab - these are formal charges of the atoms

# 2. Changed mol=Chem.RWMol(self.mol) FROM mol=self.mol in the label_tree function because we need to create a fresh mol object for assm cands. 

# 2. I have added ('H', 0) to the common atom vocab SINCE it one of the attributes of the nodes in the graph has hydrogen as its atom with formal charge zero. 

# 3. line 85 - do we need to update the clusters? in the original code, they have not updated the clusters. -NO WE DO NOT - REMOVED IT. 

# 4. I have updated the PairVocabMetal class to include the Fe in the vocab object at the end of the list of the original vocab. I am defining the tuple of (smiles,ismiles) of iron as ('Fe','Fe') 

#5. For the fnode of Fe - adding info from vocab object for iron tuple to fnode. 

#6. Fmess stores the bond type index (0-single, 1-double, 3-triple, 4-aromatic) between the nodes. the bond information between fe and other ligands should be stored in the fmess. - THIS IS AT THE ATOM LEVEL. 


# TODO

#- ADD NEW BOND TYPE FOR THE FE - NODE BONDS
# - ALSO STORE THE XYZ INFORMATION AS AN ATTRIBUTE OF THE NODES IN THE GRAPH

# LATER
# ADD ONE MORE COLUMN IN FMESS FOR STORING DISTANCE BETWEEN THE ATOMS IN THE GRAPH using the XYZ attributes stored already. 