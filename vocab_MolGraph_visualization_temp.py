# %%
from hgraph import *
import numpy as np

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

mols = MolFromSMILES()

print("Visualizing the input molecule from SMILES:")
for mol in mols:
    draw_with_spheres(mol)
# draw_with_spheres(mols[0])

# %% [markdown]
# ### a. VISUALIZING `find_clusters` FUNCTION: 1. Non-ring bonds (red) 2. Rings (green)

# %%

# import networkx as nx
# import matplotlib.pyplot as plt

# print("VISUALIZING `find_clusters` FUNCTION: 1. Non-ring bonds (red) 2. Rings (green)")

# def visualize_clusters(mol):
#     # Non-ring bonds
#     non_ring_bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds() if not bond.IsInRing()]
#     print("non-ring bonds: ", non_ring_bonds)
    
#     # Rings
#     ssr = Chem.GetSymmSSSR(mol)
#     rings_print = [list(ring) for ring in ssr]
#     print("rings: ", rings_print)
    
#     G = nx.Graph()
#     for atom in mol.GetAtoms():
#         G.add_node(atom.GetIdx(), label=atom.GetSymbol())
#     for bond in mol.GetBonds():
#         G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     pos = nx.spring_layout(G)
    
#     # Draw non-ring bonds
#     nx.draw(G, pos, ax=ax1, node_color='lightblue', node_size=500, with_labels=True, labels={node: G.nodes[node]['label'] for node in G.nodes()})
#     nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=non_ring_bonds, edge_color='r', width=2)
#     ax1.set_title('Non-ring bonds (red)')
    
#     # Draw rings
#     nx.draw(G, pos, ax=ax2, node_color='lightblue', node_size=500, with_labels=True, labels={node: G.nodes[node]['label'] for node in G.nodes()})
#     for ring in ssr:
#         ring = list(ring) 
#         ring_edges = list(zip(ring, ring[1:] + ring[:1]))
#         nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=ring_edges, edge_color='g', width=2)
#     ax2.set_title('Rings (green)')
    
#     plt.tight_layout()
#     plt.show()

# for i, mol in enumerate(mols):
#     print(f"Clusters for molecule {i+1}:")
#     visualize_clusters(mol)

# %%
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem

print("VISUALIZING `find_clusters` FUNCTION: 1. Non-ring bonds (red) 2. Rings (green)")

def visualize_clusters(mol):
    mg = MolGraph(mol)
    
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
    for i, cluster in enumerate(mg.clusters):
        cluster_center = np.mean([pos[node] for node in cluster], axis=0)
        ax1.text(cluster_center[0], cluster_center[1], f'C{i}', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        ax2.text(cluster_center[0], cluster_center[1], f'C{i}', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print("Clusters:")
    for i, cluster in enumerate(mg.clusters):
        print(f"C{i}: {cluster}")
    return mg.clusters

clusters_allmols = []
for i, mol in enumerate(mols):
    print(f"Clusters for molecule {i+1}:")
    clusters = visualize_clusters(mol)
    clusters_allmols.append(clusters)

# %%


# %%


# %% [markdown]
# ### 2. Visualizing `tree_decomp` i.e. how a graph is constructed from individual nodes (i.e. clusters)

# %%
def visualize_tree_decomp(mol, clusters):
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

    # Modified tree_decomp function to track edge types
    # clusters = mg.clusters 
    # clusters = mg.clusters#.copy() 
    graph = nx.empty_graph(len(clusters))
    
    for atom, nei_cls in enumerate(mg.atom_cls):
        if len(nei_cls) <= 1: 
            continue
        bonds = [c for c in nei_cls if len(clusters[c]) == 2]
        rings = [c for c in nei_cls if len(clusters[c]) > 4]  # need to change to 2

        print(f"Atom {atom}: nei_cls = {nei_cls}, bonds = {bonds}, rings = {rings}")
        print(f" len(clusters) = {len(clusters)}")

        if len(nei_cls) > 2 and len(bonds) >= 2:
            clusters.append([atom])
            c2 = len(clusters) - 1
            graph.add_node(c2)
            for c1 in nei_cls:
                graph.add_edge(c1, c2, weight=100, type='condition1')
            print(f"  Condition 1: Added node {c2} and edges {[(c1, c2) for c1 in nei_cls]}")
            

        elif len(rings) > 2:  # Bee Hives, len(nei_cls) > 2 
            clusters.append([atom])
            c2 = len(clusters) - 1
            graph.add_node(c2)
            for c1 in nei_cls:
                graph.add_edge(c1, c2, weight=100, type='condition2')
            print(f"  Condition 2: Added node {c2} and edges {[(c1, c2) for c1 in nei_cls]}")
        else:
            for i, c1 in enumerate(nei_cls):
                for c2 in nei_cls[i + 1:]:
                    inter = set(clusters[c1]) & set(clusters[c2])
                    graph.add_edge(c1, c2, weight=len(inter), type='condition3')
                    print(f"  Condition 3: Added edge ({c1}, {c2}) with weight {len(inter)}")

    draw_graph(graph, "Graph after all edges are added")

    draw_graph(graph, "Graph after all edges are added again")

    n, m = len(graph.nodes), len(graph.edges)
    print(f"Nodes: {n}, Edges: {m}")
    # assert n - m <= 1 #must be connected
    print("n - m == 1: ", n - m == 1)
    print("n - m <= 1: ", n - m <= 1)
    mst = nx.maximum_spanning_tree(graph)
    # graph if n - m == 1 else nx.maximum_spanning_tree(graph)
    draw_graph(mst, "Graph after applying maximum_spanning_tree")

# Visualize tree decomposition for each molecule
print(" VISUALIZING `tree_decomp` FUNCTION: 1. Red edges: condition1 2. Green edges: condition2 3. Blue edges: condition3")
for i, mol in enumerate(mols):
    print(f"Tree decomposition for molecule {i+1}:")
    visualize_tree_decomp(mol)

# %%



