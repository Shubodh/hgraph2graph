from hgraph import *
from hgraph.chemutils import *
from rdkit import Chem
from multiprocessing import Pool
import rdkit
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdDetermineBonds
import sys
import py3Dmol
import base64

current_dir = os.path.dirname(os.path.abspath(__file__))
class_dir = os.path.join(current_dir, 'hgraph')
sys.path.append(class_dir)
from mol_graph_metal import MolGraphMetal

error_molecule_ligands={}

def process(molname,mols,highlights_ligand_mol):
    vocab=set()
    if molname not in error_molecule_ligands:
        error_molecule_ligands[molname] = []

    for i,mol in enumerate(mols):
        highlight_atoms=highlights_ligand_mol[i+1]
        hmol=MolGraphMetal(mol,highlight_atoms)
        #errorhandling
        if hmol.order is None:
            print(f"Mol name {mol} is giving error with ligand{i+1} so skipping this ligand\n")
            error_molecule_ligands[molname].append(i+1)
            continue
        for node,attr in hmol.mol_tree.nodes(data=True):
            if 'smiles' not in attr:
                continue
            smiles=attr['smiles']
            vocab.add(attr['label'])
            for i,s in attr['inter_label']:
                vocab.add((smiles,s))
    return vocab

def read_molecule_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    atoms = []
    for line in lines:
        parts = line.split()
        if len(parts) == 4:
            atom_symbol, x, y, z = parts
            atoms.append((atom_symbol, float(x), float(y), float(z)))
    return atoms

def get_ligand_mols(folder_path):
    data_folder=folder_path
    ligands_map={}
    mol_ligands_obj_map={}
    mol_ligands_highlights_indexmap={}
    molecule_names=[]
    charge_array=[0,-1,-2,-3]

    for i,f in enumerate(os.listdir(data_folder)):
        file_path = os.path.join(data_folder, f)
        # print(f)
        rawmol=Chem.MolFromXYZFile(file_path)
        rdDetermineBonds.DetermineConnectivity(rawmol)
        rawmol.UpdatePropertyCache(strict=False)
        fe_atom = [atom for atom in rawmol.GetAtoms() if atom.GetSymbol() == 'Fe'][0]
        donor_atoms = [atom for atom in rawmol.GetAtoms() if rawmol.GetBondBetweenAtoms(fe_atom.GetIdx(), atom.GetIdx())]
        highlight_atoms=[]
        for atom in donor_atoms:
            highlight_atoms.append(atom.GetIdx())
        
        # function to extract ligands from the molecule by removing the metal centre.
        def get_ligand(mol, donor_atom, visited):
            """
            Identify the attached ligands in the metal chelate molecule.
            What we need to do here is identify all the ligands attached to central Metal atom i.e. Fe.

            In summary: we do BFS in one direction (the other direction than Fe). That's all to identify each ligand.

            In detail: We just found the donor atoms which are conencted to Fe. Now that donor atom will be connected to another atom say X other than Fe obviously. Now we need to start from X and find all the atoms in its connected component. So the donor atom along with all the atoms in this connected component will be one ligand. 
            
            Important-Later: **Sometimes multiple donor atoms could be in same connected component, we will have to handle this case carefully too. (see below case, we have not yet tested for bottom 3 ligands)**
            """
            ligand = []
            queue = list(donor_atom.GetNeighbors())
            
            while queue:
                current_atom = queue.pop(0)
                if current_atom.GetSymbol() != 'Fe' and current_atom.GetIdx() not in visited:
                    visited.add(current_atom.GetIdx())
                    ligand.append(current_atom)
                    queue.extend(current_atom.GetNeighbors())
            return ligand
        
        ligands = []
        visited = set()

        for i, donor_atom in enumerate(donor_atoms):
            if donor_atom.GetIdx() not in visited:
                ligand = get_ligand(rawmol, donor_atom, visited)
                ligands.append(ligand)

        atom_map={}

        # function to extract the atoms from the ligands and store them in a dictionary eg - 1-> atoms of ligand 1, 2-> atoms of ligand 2 and so on.
        for i,ligand in enumerate(ligands):
            ligand_atoms={atom.GetIdx() : atom.GetSymbol() for atom in ligand}
            atom_map[i+1]=ligand_atoms
        
        mol_name=f.split('.')[0].strip()
        # ligands_map is a dictionary where the key is the molecule  and the value is the atom_map dictionary of ligands.
        ligands_map[mol_name] = atom_map

        ligand_molblock=[]
        
        index_map={}
        # for each ligand in the ligands_map dictionary, we create a molblock for the ligand and store the old indices of the atoms in the index_map dictionary
        for ligand_number, atom_dict in ligands_map[mol_name].items():
            atom_indices = list(atom_dict.keys())
            index_map[ligand_number]=atom_indices
            data=''
            data+=f"{len(atom_dict)}\n\n"
            metaldata=read_molecule_file(os.path.join(data_folder,mol_name+'.xyz'))
            for atom_idx, atom_symbol in atom_dict.items():
                if(atom_idx < len(metaldata)):
                    atom_symbol, x, y, z = metaldata[atom_idx]
                    data+=f"{atom_symbol} {x} {y} {z}\n"
            ligand_molblock.append(data)

        # creating a new map of highlight atoms for the new ligands. each ligand entry will correspond to the atoms in the ligand that are to be highlighted.
        highlight_atoms_new={}
        lmol_array=[]

        # Now we have each individual ligand molblock and the corresponding indexes of the atoms is in index_map
        for i,ligandblock in enumerate(ligand_molblock):
            ligand_smiles = ""
            for ch in charge_array:
                lmol=Chem.MolFromXYZBlock(ligandblock)
                try:
                    rdDetermineBonds.DetermineConnectivity(lmol,charge=ch)
                    # Chem.rdmolops.CleanupOrganometallics(lmol)
                    rdDetermineBonds.DetermineBondOrders(lmol,charge=ch)
                    # print(f"charge = {ch}")
                    break
                except:
                    continue
                    # print(f"charge {ch} does not work")
            lmol.UpdatePropertyCache(strict=False)
            ligand_smiles = Chem.MolToSmiles(lmol)

            ligand_highlightatoms=[]
            partial_charges=[]

            for j,atom in enumerate(lmol.GetAtoms()):
                if(index_map[i+1][j] in highlight_atoms):
                    ligand_highlightatoms.append(atom.GetIdx())
                    # atom.SetAtomMapNum(2)
            # Chem.rdPartialCharges.ComputeGasteigerCharges(lmol)

            highlight_atoms_new[i+1]=ligand_highlightatoms
            lmol_array.append(lmol)

        mol_ligands_obj_map[mol_name]=lmol_array
        mol_ligands_highlights_indexmap[mol_name]=highlight_atoms_new
        molecule_names.append(mol_name)
    return molecule_names,mol_ligands_obj_map,mol_ligands_highlights_indexmap


if __name__=="__main__":
    # vocab_folder="data/good_full_small_vocab"
    # vocab_file="good_full_small_vocab_500.txt"
    # folderpath="data/good_full_small_500"

    vocab_folder="data/good_full_small_vocab"
    vocab_file="good_debug_vocab_5.txt"
    folderpath="data/good_debug_5"
    print("Processing for folder: ", folderpath)
    
    molecule_names,mol_ligands_obj_map,mol_ligands_highlights_indexmap = get_ligand_mols(folderpath)



    vocab_list=[]
    num=len(molecule_names)

    for i,mol in enumerate(molecule_names):
        print(mol)
        ligand_mols=mol_ligands_obj_map[mol]
        highlights_ligand_mol=mol_ligands_highlights_indexmap[mol]
        vocab=process(mol,ligand_mols,highlights_ligand_mol)
        vocab_list.append(vocab)
        print(i)
    vocab = [(x, y) for vocab_set in vocab_list for x, y in vocab_set]
    vocab = list(set(vocab))
    os.makedirs(vocab_folder, exist_ok=True)
    # Write to the file
    with open(f"{vocab_folder}/{vocab_file}", 'w') as file:
        for x, y in sorted(vocab):
            file.write(f"{x} {y}\n")
    
    # error_file="data/good_full_small_vocab/error.txt"
    # with open(error_file, 'w') as file:
    #     for molname, error_ligands in error_molecule_ligands.items():
    #         # Check if molecule name exists in mol_ligands_object_map
    #         if molname not in mol_ligands_obj_map:
    #             file.write(f"Molecule '{molname}' not found in mol_ligands_obj_map\n")
    #             continue
            
    #         # Retrieve the SMILES map for this molecule
    #         ligand_smiles_map = mol_ligands_obj_map[molname]
            
    #         file.write(f"Errors for molecule: {molname}\n")
    #         file.write("-" * 40 + "\n")
            
    #         # Retrieve and print SMILES for each error ligand ID
    #         for error_id in error_ligands:
    #             smiles = get_smiles(ligand_smiles_map[error_id])
    #             file.write(f"Ligand ID: {error_id}, SMILES: {smiles}\n")
            
    #         file.write("\n")  # Add a newline for readability between different molecules




