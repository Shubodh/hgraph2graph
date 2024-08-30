# %%
# from hgraph import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

# %%
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import py3Dmol
from IPython.display import display

# Function to visualize molecule using py3Dmol
def draw_with_spheres(mol):
    v = py3Dmol.view(width=300,height=300)
    mol_block = Chem.MolToMolBlock(mol)
    v.addModel(mol_block, "mol")
    v.zoomTo()
    v.setStyle({'sphere':{'radius':0.3},'stick':{'radius':0.2}})
    v.show()

# Function to load molecules from SMILES strings
def MolFromSMILES():
    smiles_list = [
        "CSc1ccccc1C(=O)Nc1ccc2c(c1)OCCO2",
        # "CC(C)NC(=N)c1ccc(OCCCCCOc2ccc(C(=N)NC(C)C)cc2C(=O)O)c(C(=O)O)c1",
        # "O=C(Cn1cn[nH]c1=O)N1CC2(CCCC2)c2ccccc21"
    ]
    mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        mols.append(mol)
    return mols

# mols = MolFromSMILES()

# print("Visualizing the input molecule from SMILES:")
# for mol in mols:
#     draw_with_spheres(mol)
# draw_with_spheres(mols[0])

# %%
def build_mol_graph(mol):
    graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
    for atom in mol.GetAtoms():
        graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        btype = BOND_LIST.index(bond.GetBondType())
        graph[a1][a2]['label'] = btype
        graph[a2][a1]['label'] = btype

    return graph

def find_clusters(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append((a1, a2))

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    if clusters and 0 not in clusters[0]:
        for i, cls in enumerate(clusters):
            if 0 in cls:
                clusters = [clusters[i]] + clusters[:i] + clusters[i+1:]
                break

    atom_cls = [[] for _ in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls

def tree_decomp(clusters, atom_cls):
    graph = nx.Graph()
    for i in range(len(clusters)):
        graph.add_node(i)

    for atom, nei_cls in enumerate(atom_cls):
        if len(nei_cls) <= 1:
            continue
        bonds = [c for c in nei_cls if len(clusters[c]) == 2]
        rings = [c for c in nei_cls if len(clusters[c]) > 4]

        if len(nei_cls) > 2 and len(bonds) >= 2:
            c2 = len(clusters)
            clusters.append([atom])
            graph.add_node(c2)
            for c1 in nei_cls:
                graph.add_edge(c1, c2, weight=100, type='condition1')

        elif len(rings) > 2:
            c2 = len(clusters)
            clusters.append([atom])
            graph.add_node(c2)
            for c1 in nei_cls:
                graph.add_edge(c1, c2, weight=100, type='condition2')

        else:
            for i, c1 in enumerate(nei_cls):
                for c2 in nei_cls[i + 1:]:
                    inter = set(clusters[c1]) & set(clusters[c2])
                    graph.add_edge(c1, c2, weight=len(inter), type='condition3')

    return graph, clusters

# %% [markdown]
# ### a. VISUALIZING `find_clusters` FUNCTION: 1. Non-ring bonds (red) 2. Rings (green)

# %%

print("VISUALIZING `find_clusters` FUNCTION: 1. Non-ring bonds (red) 2. Rings (green)")

def visualize_clusters(mol):
    clusters, atom_cls = find_clusters(mol)
    
    # Non-ring bonds
    non_ring_bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds() if not bond.IsInRing()]
    print("non-ring bonds: ", non_ring_bonds)
    
    # Rings
    ssr = Chem.GetSymmSSSR(mol)
    rings_print = [list(ring) for ring in ssr]
    print("rings: ", rings_print)
    
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=f"{atom.GetSymbol()}{atom.GetIdx()}")
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.spring_layout(G)
    
    # Draw non-ring bonds
    nx.draw(G, pos, ax=ax1, node_color='lightblue', node_size=700, with_labels=True, labels={node: G.nodes[node]['label'] for node in G.nodes()})
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=non_ring_bonds, edge_color='r', width=2)
    ax1.set_title('Non-ring bonds (red)')
    
    # Draw rings
    nx.draw(G, pos, ax=ax2, node_color='lightblue', node_size=700, with_labels=True, labels={node: G.nodes[node]['label'] for node in G.nodes()})
    for ring in ssr:
        ring = list(ring) 
        ring_edges = list(zip(ring, ring[1:] + ring[:1]))
        nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=ring_edges, edge_color='g', width=2)
    ax2.set_title('Rings (green)')
    
    # Add cluster IDs
    for i, cluster in enumerate(clusters):
        cluster_center = np.mean([pos[node] for node in cluster], axis=0)
        ax1.text(cluster_center[0], cluster_center[1], f'c:{i}', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        ax2.text(cluster_center[0], cluster_center[1], f'c:{i}', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print("Clusters:")
    for i, cluster in enumerate(clusters):
        print(f"c:{i} -- {cluster}")
    return clusters, atom_cls

# clusters_allmols = []
# atom_cls_all = []
# for i, mol in enumerate(mols):
#     print(f"Clusters for molecule {i+1}:")
#     clusters, atom_cls = visualize_clusters(mol)
#     clusters_allmols.append(clusters)
#     atom_cls_all.append(atom_cls)

# %%


# %%


# %% [markdown]
# ### 2. Visualizing `tree_decomp` i.e. how a graph is constructed from individual nodes (i.e. clusters)

# %%
def visualize_tree_decomp(mol, clusters, atom_cls):
    # mg = MolGraph(mol)
    
    def draw_graph(G, title):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        nx.draw_networkx_nodes(G, pos, node_size=[len(clusters[n])*100 for n in G.nodes()])
        
        # edges with different colors based on their type
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 'condition1'], edge_color='r', width=2)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 'condition2'], edge_color='g', width=2)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 'condition3'], edge_color='b', width=2)
        
        labels = {i: f"{i}: {clusters[i]}" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    graph, updated_clusters = tree_decomp(clusters, atom_cls)
    draw_graph(graph, "Graph after all edges are added")

    draw_graph(graph, "Graph after all edges are added again")

    n, m = len(graph.nodes), len(graph.edges)
    print(f"Nodes: {n}, Edges: {m}")
    assert n - m <= 1 #must be connected
    print("n - m == 1: ", n - m == 1)
    print("n - m <= 1: ", n - m <= 1)
    if n - m == 1:
        print("Graph is already a tree, no further visualization needed")
        mst = graph
    else:
        mst = nx.maximum_spanning_tree(graph)
        draw_graph(mst, "Graph after applying maximum_spanning_tree")
    # graph if n - m == 1 else nx.maximum_spanning_tree(graph)
    return mst, updated_clusters

# # Visualize tree decomposition for each molecule
# print(" VISUALIZING `tree_decomp` FUNCTION: 1. Red edges: condition1 2. Green edges: condition2 3. Blue edges: condition3")
# for i, mol in enumerate(mols):
#     print(f"Tree decomposition for molecule {i+1}:")
#     visualize_tree_decomp(mol)

# %%


# %%


# %% [markdown]
# # MAIN

# %%
mols = MolFromSMILES()

print("Visualizing the input molecule from SMILES:")
for mol in mols:
    draw_with_spheres(mol)

# %%
# mol_graphs = build_mol_graph(mols[0])
mol_graphs = []
for i, mol in enumerate(mols):
	mol_graph = build_mol_graph(mol)
	mol_graphs.append(mol_graph)

# %% [markdown]
# ### a. Visualizing `find_clusters` function: 1. Non-ring bonds (red) 2. Rings (green)

# %%
print("VISUALIZING `find_clusters` FUNCTION: 1. Non-ring bonds (red) 2. Rings (green)")
clusters_allmols = []
atom_cls_allmols = []
for i, mol in enumerate(mols):
    print(f"Clusters for molecule {i+1}:")
    clusters, atom_cls = visualize_clusters(mol)
    clusters_allmols.append(clusters)
    atom_cls_allmols.append(atom_cls)

# %% [markdown]
# ### b. VISUALIZING `tree_decomp` FUNCTION: 1. red edges: condition1 2. green edges: condition2 3. blue edges: condition3 

# %% [markdown]
#  **`tree_decomp` : how graph is constructed from individual nodes (aka clusters)** -
# 
# TODO: graph vs tree where happening exactly here, mst here, clarify later. Shouldn't a graph be created before mst is applied as per your understanding ? why are you getting tree directly?
# 
# For every atom that is shared between at least 2 neighboring clusters:
# 
# 1. If the atom is shared by more than 2 clusters, and at least 2 of these clusters are bonds (clusters of size 2):
#     - A new cluster is created containing just this atom.
#     - Edges are added between this new cluster and all the clusters that contain this atom.
#     - These edges are given a high weight (100).
# 2. If the atom is shared by more than 2 rings (clusters of size > 4):
#     - Again, a new cluster is created containing just this atom.
#     - Edges are added between this new cluster and all the clusters that contain this atom.
#     - These edges are also given a high weight (100).
# 3. If neither of the above conditions are met:
#     - Edges are added between all pairs of clusters that contain this atom.
#     - The weight of each edge is set to the number of atoms shared between the two clusters.

# %%
print("VISUALIZING `tree_decomp` FUNCTION: 1. Red edges: condition1 2. Green edges: condition2 3. Blue edges: condition3")
trees_allmols = []
updated_clusters_all_mols = []
for i, mol in enumerate(mols):
    print(f"Tree decomposition for molecule {i+1}:")
    tree, updated_clusters =  visualize_tree_decomp(mol, clusters_allmols[i], atom_cls_allmols[i])
    trees_allmols.append(tree)
    # clusters_allmols[i] = updated_clusters # wrong
    updated_clusters_all_mols.append(updated_clusters)

# IMPORTANT: Replacing the clusters with updated clusters
clusters_allmols = updated_clusters_all_mols

# %%


# %%


# %% [markdown]
# ### c. VISUALIZING `label_tree`: It is complex than before so we will visualize it in parts as follows:
# 
# 

# %% [markdown]
# 1. Depth-First Search (DFS) Traversal
# 	- Visualize the order in which nodes are visited
# 	- Show parent-child relationships
# 	- Illustrate the assignment of labels to edges
# 
# 2. Inter-atom labeling
# 	- Show how inter-atoms are identified
# 	- Visualize the process of creating inter-labels
# 
# 3. Atom mapping
# 	- Display how atoms are mapped in the original molecule
# 
# 4. Cluster and inter-label assignment
# 	- Visualize the assignment of SMILES and inter-labels to each cluster
# 
# 5. Assembly candidates
# 	- Show how assembly candidates are determined for each cluster
# 

# %% [markdown]
# #### core functions

# %%
from hgraph.chemutils import get_clique_mol, is_anchor, get_anchor_smiles, set_atommap
from hgraph.chemutils import get_smiles

idxfunc = lambda a : a.GetAtomMapNum() - 1

# %%
def get_inter_label(mol, atoms, inter_atoms):
    new_mol = get_clique_mol(mol, atoms)
    
    if new_mol.GetNumBonds() == 0:
        inter_atom = list(inter_atoms)[0]
        for a in new_mol.GetAtoms():
            a.SetAtomMapNum(0)
        return new_mol, [(inter_atom, Chem.MolToSmiles(new_mol))]
    
    inter_label = []
    for a in new_mol.GetAtoms():
        idx = idxfunc(a)
        if idx in inter_atoms and is_anchor(a, inter_atoms):
            inter_label.append((idx, get_anchor_smiles(new_mol, idx)))

    for a in new_mol.GetAtoms():
        a.SetAtomMapNum( 1 if idxfunc(a) in inter_atoms else 0 )
    
    return new_mol, inter_label


def get_assm_cands(mol, atoms, inter_label, cluster, inter_size):
    atoms = list(set(atoms))
    mol = get_clique_mol(mol, atoms)
    atom_map = [idxfunc(atom) for atom in mol.GetAtoms()]
    mol = set_atommap(mol)
    rank = Chem.CanonicalRankAtoms(mol, breakTies=False)
    rank = { x:y for x,y in zip(atom_map, rank) }

    pos, icls = zip(*inter_label)
    if inter_size == 1:
        cands = [pos[0]] + [ x for x in cluster if rank[x] != rank[pos[0]] ] 
    
    elif icls[0] == icls[1]: #symmetric case
        shift = cluster[inter_size - 1:] + cluster[:inter_size - 1]
        cands = zip(cluster, shift)
        cands = [pos] + [ (x,y) for x,y in cands if (rank[min(x,y)],rank[max(x,y)]) != (rank[min(pos)], rank[max(pos)]) ]
    else: 
        shift = cluster[inter_size - 1:] + cluster[:inter_size - 1]
        cands = zip(cluster + shift, shift + cluster)
        cands = [pos] + [ (x,y) for x,y in cands if (rank[x],rank[y]) != (rank[pos[0]], rank[pos[1]]) ]

    return cands



# %%

def label_tree(mol, tree, clusters, mol_graph):
    def dfs(order, pa, prev_sib, x, fa):
        pa[x] = fa
        sorted_child = sorted([y for y in tree[x] if y != fa])
        for idx, y in enumerate(sorted_child):
            tree[x][y]['label'] = 0
            tree[y][x]['label'] = idx + 1  # position encoding
            prev_sib[y] = sorted_child[:idx]
            prev_sib[y] += [x, fa] if fa >= 0 else [x]
            order.append((x, y, 1))
            dfs(order, pa, prev_sib, y, x)
            order.append((y, x, 0))

    order, pa = [], {}
    tree = nx.DiGraph(tree)
    prev_sib = [[] for _ in range(len(clusters))]
    dfs(order, pa, prev_sib, 0, -1)
    order.append((0, None, 0))  # last backtrack at root

    # TODO: Should you recreate mol object? See original code which does it from smiles

    # mol = get_mol(smiles)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)

    for i, cls in enumerate(clusters):
        inter_atoms = set(cls) & set(clusters[pa[i]]) if pa[i] >= 0 else set([0])
        cmol, inter_label = get_inter_label(mol, cls, inter_atoms)
        tree.nodes[i]['ismiles'] = ismiles = get_smiles(cmol)
        tree.nodes[i]['inter_label'] = inter_label
        tree.nodes[i]['smiles'] = smiles = get_smiles(set_atommap(cmol))
        tree.nodes[i]['label'] = (smiles, ismiles) if len(cls) > 1 else (smiles, smiles)
        tree.nodes[i]['cluster'] = cls
        tree.nodes[i]['assm_cands'] = []

        if pa[i] >= 0 and len(clusters[pa[i]]) > 2:  # uncertainty occurs in assembly
            hist = [a for c in prev_sib[i] for a in clusters[c]]
            pa_cls = clusters[pa[i]]
            tree.nodes[i]['assm_cands'] = get_assm_cands(mol, hist, inter_label, pa_cls, len(inter_atoms))

            child_order = tree[i][pa[i]]['label']
            diff = set(cls) - set(pa_cls)
            for fa_atom in inter_atoms:
                for ch_atom in mol_graph[fa_atom]:
                    if ch_atom in diff:
                        label = mol_graph[ch_atom][fa_atom]['label']
                        if type(label) is int: #in case one bond is assigned multiple times
                            mol_graph[ch_atom][fa_atom]['label'] = (label, child_order)

    return order, tree, mol_graph

# %%


# %% [markdown]
# #### visualization functions

# %% [markdown]
# 

# %%
# TODO: Add visualization functions here
def visualize_dfs_traversal(mol, labeled_tree, order):
	pass

# %% [markdown]
# #### MAIN C. visualization

# %%
def visualize_label_tree(mol, tree, clusters, mol_graph):
    order, labeled_tree, mol_graph = label_tree(mol, tree, clusters, mol_graph)
    
    visualize_dfs_traversal(mol, labeled_tree, order)
    # visualize_inter_atom_labeling(mol, labeled_tree, clusters)
    # visualize_atom_mapping(mol)
    # visualize_cluster_assignments(mol, labeled_tree, clusters)
    # visualize_assembly_candidates(mol, labeled_tree, clusters)

# %%

print("VISUALIZING `label_tree` FUNCTION:")
labeled_trees_allmols = []
for i, mol in enumerate(mols):
    print(f"\nLabeling tree for molecule {i+1}:")
    visualize_label_tree(mol, trees_allmols[i], clusters_allmols[i], mol_graphs[i])
 

# %%



