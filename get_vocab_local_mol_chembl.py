# import sys
# import argparse 
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool

def process_old(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraphMetal(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab

def process(mols):
    vocab = set()
    for num, mol in enumerate(mols):
        print(f"PROCESSING molecule {num} with SMILES")
        hmol = MolGraphMetal(mol)
        for node, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add(attr['label'])
            for i, s in attr['inter_label']:
                vocab.add((smiles, s))
    return vocab

def MolFromSMILES():
    smiles_list = [
        "CSc1ccccc1C(=O)Nc1ccc2c(c1)OCCO2",
        # "CC(C)NC(=N)c1ccc(OCCCCCOc2ccc(C(=N)NC(C)C)cc2C(=O)O)c(C(=O)O)c1",
        # "O=C(Cn1cn[nH]c1=O)N1CC2(CCCC2)c2ccccc21"

        # "CSc1ccccc1C(=O)Nc1ccc2c(c1)OCCO2",
        # "CC(C)NC(=N)c1ccc(OCCCCCOc2ccc(C(=N)NC(C)C)cc2C(=O)O)c(C(=O)O)c1",
        # "O=C(Cn1cn[nH]c1=O)N1CC2(CCCC2)c2ccccc21",
        # "CCN1CCCN(C(=O)c2ccc(-c3[nH]nc4c3C(=O)c3c(NC(=O)NN5CCOCC5)cccc3-4)s2)CC1",
        # "COCCOP(=O)(OCCOC)C(NC(SC)=C(C#N)C(=O)OC)c1ccccc1F",
        # "CC(C)N1CCc2c(sc(NC(=O)C(F)F)c2-c2nc3ccccc3s2)C1",
        # "Cc1ccc(N(C)C(=O)C2CCCN2S(=O)(=O)c2cccc3nsnc23)cc1",
        # "Cc1ccc2nc(N(C)C)nc(NCc3ccc(C(=O)Nc4ccc(Cl)nc4)cc3)c2c1",
        # "OCCNc1ncnc2c1ncn2C1CN(Cc2cccs2)CC(CO)O1",
        # "COc1ccc(CNC(=O)CSc2nc3c(c(=O)n(C)c(=O)n3C)n2C)cc1",

        # "CCOC(=O)C=C1NC(=O)C1C(C)OC(=O)c1cc(OCc2ccccc2)c(OCc2ccccc2)c(OCc2ccccc2)c1",
        # "c1ccc(C2=Nn3cnnc3SC2)cc1",
        # "CC(C)OC(=O)c1c(N=c2scc(-c3ccccc3)n2-c2ccccc2)sc2c1CCCC2",
        # "Nc1ncc(-c2cnn(C3CCC(O)CC3)c2)c2c(CF)c(-c3cccc4nnsc34)oc12",
        # "CC1(C(=O)OCC(=O)NCCNC(=O)COC(=O)C2(C)CC2(Cl)Cl)CC1(Cl)Cl",
        # "O=C(NO)c1ccc(C(=O)NN=Cc2ccc(OCCCCCCCCCOc3ccc(C=NNC(=O)c4ccc(C(=O)NO)cc4)cc3)cc2)cc1",
        # "O=C(NN=Cc1ccc([N+](=O)[O-])cc1)c1cc(OCc2ccccc2)c(OCc2ccccc2)c(OCc2ccccc2)c1",
        # "Cn1c(CC(=O)Nc2ccc(F)cc2)nc(N2CCOCC2)cc1=O",
        # "Cc1noc(C)c1C(=O)N1CCC(Cc2ccccc2)CC1",
        # "CN(C)c1ccc(-c2cn3cc(Cl)ncc3n2)cc1",
        # "COc1ccc2c(c1)SCc1cnc(-c3ccc(Cl)cc3)nc1-2",
        # "Nc1[nH]c(C(=O)c2ccccc2)c(-c2ccncc2)c1C(=O)NCc1cccc2ccccc12",
        # "OCc1ccc(-c2nccn2-c2cccc3c2CCC3)o1",
        # "COc1cccc(C(=O)OC2CCC3(C)C(CCC4(C)C3CC=C3C5CC(C)(C)CC(OC(=O)C=C(C)C)C5(C(=O)O)CCC34C)C2(C)C)c1",
        # "COc1ccc(C(=O)NC(=S)N2CCCCC2c2cccnc2)cc1OC",
        # "CCC(=O)N(C)CC1Oc2cc(C#CC3CCCC3)ccc2S(=O)(=O)N(C(C)CO)CC1C",

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
    vocab_file = 'vocab_small_MolObj_rand.txt'


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
