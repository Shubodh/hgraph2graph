# import sys
# import argparse 
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool

def process_old(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab

def process(mols):
    vocab = set()
    for mol in mols:
        hmol = MolGraph(mol)
        for node, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add(attr['label'])
            for i, s in attr['inter_label']:
                vocab.add((smiles, s))
    return vocab

def MolFromSMILES():
    smiles_list = [
        "CSc1ccccc1C(=O)Nc1ccc2c(c1)OCCO2",
        "CC(C)NC(=N)c1ccc(OCCCCCOc2ccc(C(=N)NC(C)C)cc2C(=O)O)c(C(=O)O)c1",
        "O=C(Cn1cn[nH]c1=O)N1CC2(CCCC2)c2ccccc21"
    ]
    mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        mols.append(mol)
    return mols

def MolLigandsFromFile():
    pass

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--ncpu', type=int, default=1)
    # args = parser.parse_args()
    ncpu = 16
    mols = MolFromSMILES()
    print("Processing for mols: ", mols)


    data_folder = 'data/chembl/'
    vocab_folder = 'data/chembl/vocab/'

    data_file = 'all_small.txt'
    vocab_file = 'vocab_small_MolObj.txt'


    # with open(f"{data_folder}/{data_file}", 'r') as file:
    #     data = [mol for line in file for mol in line.split()[:2]]
    # data = list(set(data))

    # batch_size = len(data) // ncpu + 1
    # batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    # pool = Pool(ncpu)
    # vocab_list = pool.map(process, batches)
    vocab_list = [process(mols)]
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    for x,y in sorted(vocab):
        print(x, y)

    # Write output to file
    with open(f"{vocab_folder}/{vocab_file}", 'w') as file:
        for x, y in sorted(vocab):
            file.write(f"{x} {y}\n")
