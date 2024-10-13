# python3 preprocess_metal.py --train_folder data/small_3/molecules/ --vocab data/small_3/small_3.txt 

from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy
import os

from hgraph import MolGraphMetal, common_atom_vocab_metal, PairVocabMetal
import rdkit


def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

# DO NOT iterate over complexes names when the total data size is lesser than the pool that ur using because then it will iterate over the characters instead of the molecules in total. 

def tensorize_metal(complexes_names,complexes_ligands, complexes_highlights,vocab):
    x = MolGraphMetal.tensorize_metal(complexes_names, complexes_ligands, complexes_highlights, vocab, common_atom_vocab_metal)
    return to_numpy(x)

def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraphMetal.tensorize(x, vocab, common_atom_vocab_metal)
    y = MolGraphMetal.tensorize(y, vocab, common_atom_vocab_metal)
    return to_numpy(x)[:-1] + to_numpy(y) #no need of order for x

def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = numpy.array(cond)
    x = MolGraphMetal.tensorize(x, vocab, common_atom_vocab_metal)
    y = MolGraphMetal.tensorize(y, vocab, common_atom_vocab_metal)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,) #no need of order for x

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
        rawmol = Chem.MolFromXYZFile(file_path)
        rdDetermineBonds.DetermineConnectivity(rawmol)
        rawmol.UpdatePropertyCache(strict=False)
        fe_atom=[atom for atom in rawmol.GetAtoms() if atom.GetSymbol()=='Fe'][0]
        donor_atoms=[atom for atom in rawmol.GetAtoms() if rawmol.GetBondBetweenAtoms(fe_atom.GetIdx(), atom.GetIdx())]
        highlight_atoms=[]
        for atom in donor_atoms:
            highlight_atoms.append(atom.GetIdx())
        
        def get_ligand(mol,donor_atom,visited):
            ligand=[]
            queue=list(donor_atom.GetNeighbors())
            while queue:
                current_atom=queue.pop(0)
                if current_atom.GetSymbol() != 'Fe' and current_atom.GetIdx() not in visited:
                    visited.add(current_atom.GetIdx())
                    ligand.append(current_atom)
                    queue.extend(current_atom.GetNeighbors())
            return ligand

        ligands=[]
        visited=set()

        for i,donor_atom in enumerate(donor_atoms):
            if donor_atom.GetIdx() not in visited:
                ligand=get_ligand(rawmol,donor_atom,visited)
                ligands.append(ligand)
        
        atom_map={}

        for i,ligand in enumerate(ligands):
            ligand_atoms={atom.GetIdx():atom.GetSymbol() for atom in ligand}
            atom_map[i+1]=ligand_atoms
        
        mol_name=f.split('.')[0].strip()
        ligands_map[mol_name]=atom_map

        ligand_molblock=[]
        index_map={}

        for ligand_number,atom_dict in ligands_map[mol_name].items():
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
        

        highlight_atoms_new={}
        lmol_array=[]

        for i,ligandblock in enumerate(ligand_molblock):
            ligand_smiles=""
            for ch in charge_array:
                lmol=Chem.MolFromXYZBlock(ligandblock)
                try:
                    rdDetermineBonds.DetermineConnectivity(lmol,charge=ch)
                    rdDetermineBonds.DetermineBondOrders(lmol,charge=ch)
                    break
                except:
                    continue
            lmol.UpdatePropertyCache(strict=False)
            ligand_smiles=Chem.MolToSmiles(lmol)

            ligand_highlightatoms=[]

            for j,atom in enumerate(lmol.GetAtoms()):
                if(index_map[i+1][j] in highlight_atoms):
                    ligand_highlightatoms.append(j)
            
            highlight_atoms_new[i+1]=ligand_highlightatoms
            lmol_array.append(lmol)
        
        mol_ligands_obj_map[mol_name]=lmol_array
        mol_ligands_highlights_indexmap[mol_name]=highlight_atoms_new
        molecule_names.append(mol_name)
    
    return molecule_names,mol_ligands_obj_map, mol_ligands_highlights_indexmap

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL) # used to log messages about the operation of the RDKit library

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--batch_size', type=int, default=4) # changed from 32 to 4 
    parser.add_argument('--mode', type=str, default='single') # changed from 'pair' to 'single'
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    args.vocab = PairVocabMetal(vocab, cuda=False)

    pool = Pool(args.ncpu) 
    random.seed(1)
    
    if args.mode == 'single':
        print("Single mode")
        #dataset contains single molecules

        complexes_names,complexes_ligands,complexes_highlights = get_ligand_mols(args.train_folder)
        print("Molecule names: ", complexes_names)
        

        # --------- USE THIS FOR SLICING BATCHING THE DATA LATER WITH BIGGER DATASETS----------------
        # mapping=dict(list(complexes_ligands.items())[:2])
        # names=complexes_names[:2]
        # for mol in names:
        #     print(mol)
        #     print(len(mapping[mol]))
        # --------------------------------------------------------------------------------------------

        # --------- USE THIS FOR POOLING THE DATA WITH MULTIPLE PROCESSES LATER----------------
        # func = partial(tensorize_metal, vocab = args.vocab, complexes_ligands=complexes_ligands)
        # all_data = pool.map(func, complexes_names)

        small_data = tensorize_metal(complexes_names=complexes_names,complexes_ligands=complexes_ligands, complexes_highlights=complexes_highlights,vocab=args.vocab)
        print("small data recieved")
        print(len(small_data))

        with open('data/small_3_tensor/small_data.pkl', 'wb') as file:
            pickle.dump(small_data, file,pickle.HIGHEST_PROTOCOL)


# +3, +2, +1, 0 - For FE atoms
        