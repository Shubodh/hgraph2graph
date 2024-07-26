# import sys
# import argparse 
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool

def process(data):
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

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--ncpu', type=int, default=1)
    # args = parser.parse_args()
    ncpu = 16
    data_folder = 'data/chembl/'
    vocab_folder = 'data/chembl/vocab/'

    data_file = 'all_small.txt'
    vocab_file = 'vocab_small.txt'

    with open(f"{data_folder}/{data_file}", 'r') as file:
        data = [mol for line in file for mol in line.split()[:2]]
    data = list(set(data))

    batch_size = len(data) // ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    for x,y in sorted(vocab):
        print(x, y)

    # Write output to file
    with open(f"{vocab_folder}/{vocab_file}", 'w') as file:
        for x, y in sorted(vocab):
            file.write(f"{x} {y}\n")
