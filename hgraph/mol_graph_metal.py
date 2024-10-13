import torch
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from hgraph.chemutils import *
from hgraph.nnutils import *

add = lambda x,y : x + y if type(x) is int else (x[0] + y, x[1] + y)

class MolGraphMetal(object):

    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 20

    def __init__(self, mol,highlight_atoms):
        # self.smiles = smiles
        # self.mol = get_mol(smiles)
        self.mol = mol
        self.highlight=highlight_atoms
        self.mol_graph = self.build_mol_graph()
        self.clusters, self.atom_cls = self.find_clusters()
        self.mol_tree = self.tree_decomp()
        self.order = self.label_tree()

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
        clusters.extend(ssr)

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

    def tree_decomp(self):
        clusters_tree = self.clusters
        graph=nx.Graph()
        for i in range(len(clusters_tree)):
            graph.add_node(i)
        
        for atom, nei_cls in enumerate(self.atom_cls):
            if len(nei_cls) <= 1: continue
            bonds = [c for c in nei_cls if len(clusters_tree[c]) == 2]
            rings = [c for c in nei_cls if len(clusters_tree[c]) > 4]

            if len(nei_cls) > 2 and len(bonds) >= 2:
                clusters_tree.append([atom])
                c2 = len(clusters_tree)-1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)

            elif len(rings) > 2: #Bee Hives, len(nei_cls) > 2 
                clusters_tree.append([atom]) #temporary value, need to change
                c2 = len(clusters_tree)-1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)
            else:
                for i,c1 in enumerate(nei_cls):
                    for c2 in nei_cls[i + 1:]:
                        inter = set(clusters_tree[c1]) & set(clusters_tree[c2])
                        graph.add_edge(c1, c2, weight = len(inter))

        self.clusters=clusters_tree
        n, m = len(graph.nodes), len(graph.edges)
        if n-m==1:
            mst=graph
        else:
            mst=nx.maximum_spanning_tree(graph) #must be connected
        return mst

    def label_tree(self):
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
        
        # TODO: Molecule recreation: new clean mol object might be necessary here (to ensure
        # fresh atom map numbers), so using self.mol might be wrong. Need to correct outer code accordingly. 
        # mol = get_mol(self.smiles) # modified this to remove smiles input dependency completely
        mol = self.mol #added
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

            mol=self.mol
            # if pa[i] >= 0 and len(self.clusters[ pa[i] ]) > 2: #uncertainty occurs in assembly
            #     hist = [a for c in prev_sib[i] for a in self.clusters[c]] 
            #     pa_cls = self.clusters[pa[i]]
            #     cands = get_assm_cands(mol, hist, inter_label, pa_cls, len(inter_atoms)) 
                
            #     #errorhandling
            #     #-------------------------------------
            #     if cands is not None:
            #         tree.nodes[i]['assm_cands']=cands
            #     else:
            #         continue
                    
            #     #--------------------------------------

            #     child_order = tree[i][pa[i]]['label']
            #     diff = set(cls) - set(pa_cls)
            #     for fa_atom in inter_atoms:
            #         for ch_atom in self.mol_graph[fa_atom]:
            #             if ch_atom in diff:
            #                 label = self.mol_graph[ch_atom][fa_atom]['label']
            #                 if type(label) is int: #in case one bond is assigned multiple times
            #                     self.mol_graph[ch_atom][fa_atom]['label'] = (label, child_order)
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
    
    @staticmethod
    def tensorize_metal(complexes_names_batch, complexes_ligands, complexes_highlights, vocab, avocab):
        mol_batch_metal_map = {}
        mol_tree_metal_map = {}
        mol_graph_metal_map = {}

        for idx, complex in enumerate(complexes_names_batch):
            ligands_set = complexes_ligands[complex]
            ligands_highlights = complexes_highlights[complex]

            mol_batch_metal_map[complex] = [MolGraphMetal(ligand,ligands_highlights[i+1]) for i,ligand in enumerate(ligands_set)]

            mol_tree_metal_map[complex] = [x.mol_tree for x in mol_batch_metal_map[complex]]
            mol_graph_metal_map[complex] = [x.mol_graph for x in mol_batch_metal_map[complex]]
        
        tree_tensors, tree_batchG = MolGraphMetal.tensorize_graph_metal(complexes_names_batch, mol_tree_metal_map, vocab)
        graph_tensors, graph_batchG = MolGraphMetal.tensorize_graph_metal(complexes_names_batch, mol_graph_metal_map, avocab,('Fe',0), use_highlights=True)

        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1] 

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
    
    def tensorize_graph_metal(complexes_names_batch, mol_metal_map_batch, vocab, iron_tuple=('Fe', 'Fe:2'), use_highlights=False):
        fnode, fmess = [None], [(0, 0, 0, 0)]
        agraph, bgraph = [[]], [[]]
        scope = []
        edge_dict = {}
        all_G = []

        for bid, complex_name in enumerate(complexes_names_batch):
            graphs = mol_metal_map_batch[complex_name]

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

                if use_highlights:
                    for v, attr in G.nodes(data='highlight'):
                        if attr == 1:  # Check if the node is highlighted
                            fmess.append((iron_index, v, 0, 0))  # Default bond type
                            edge_dict[(iron_index, v)] = eid = len(edge_dict) + 1
                            agraph[v].append(eid)
                            bgraph.append([])
                else:
                    for v, attr in G.nodes(data='label'):
                        if ':2' in attr[1]:  # Check if the label contains ':2'
                            fmess.append((iron_index, v, 0, 0))  # Default bond type
                            edge_dict[(iron_index, v)] = eid = len(edge_dict) + 1
                            agraph[v].append(eid)
                            bgraph.append([])

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

        fnode[0] = fnode[1]  # Set the first node to the iron node
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)

        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)
        
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

# 2. I have added ('H', 0) to the common atom vocab SINCE it one of the attributes of the nodes in the graph has hydrogen as its atom with formal charge zero. 

# 3. line 85 - do we need to update the clusters? in the original code, they have not updated the clusters.

# 4. I have updated the PairVocabMetal class to include the Fe in the vocab object at the end of the list of the original vocab. I am defining the tuple of (smiles,ismiles) of iron as ('Fe','Fe')

#5. For the fnode of Fe - adding info from vocab object for iron tuple to fnode. 

#6. Fmess stores the bond type index (0-single, 1-double, 3-triple, 4-aromatic) between the nodes. the bond information between fe and other ligands should be stored in the fmess.


# TODO

#- ADD NEW BOND TYPE FOR THE FE - NODE BONDS
# - ALSO STORE THE XYZ INFORMATION AS AN ATTRIBUTE OF THE NODES IN THE GRAPH

# LATER
# ADD ONE MORE COLUMN IN FMESS FOR STORING DISTANCE BETWEEN THE ATOMS IN THE GRAPH.