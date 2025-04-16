import rdkit
import rdkit.Chem as Chem
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

idxfunc = lambda a : a.GetAtomMapNum() - 1

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
    return mol

def get_mol(smiles): # sanitizing is being done here while retrieving the molecule object from smiles string.
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def is_aromatic_ring(mol):
    if mol.GetNumAtoms() == mol.GetNumBonds(): 
        aroma_bonds = [b for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        return len(aroma_bonds) == mol.GetNumBonds()
    else:
        return False

def get_leaves(mol):
    leaf_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( set([a1,a2]) )

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(rings)

    leaf_rings = []
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        if len(inters) > 1: continue
        nodes = [i for i in r if mol.GetAtomWithIdx(i).GetDegree() == 2]
        leaf_rings.append( max(nodes) )

    return leaf_atoms + leaf_rings

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def bond_match(mol1, a1, b1, mol2, a2, b2):
    a1,b1 = mol1.GetAtomWithIdx(a1), mol1.GetAtomWithIdx(b1)
    a2,b2 = mol2.GetAtomWithIdx(a2), mol2.GetAtomWithIdx(b2)
    return atom_equal(a1,a2) and atom_equal(b1,b2)

def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

#mol must be RWMol object
def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

def get_sub_mol_metal(mol, sub_atoms, mol_bonds):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        #if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol

def get_clique_mol(mol, atoms):
    #errorhandling
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = copy_edit_mol(new_mol).GetMol()
        new_mol = sanitize(new_mol) 
        #if tmp_mol is not None: new_mol = tmp_mol
        return new_mol
    except:
        return None

def get_clique_mol_new(mol,atoms,highlights):
    #errorhandling
    try:
        atom_maps=[] #
        for a in mol.GetAtoms(): #
            if a.GetIdx() in atoms: #
                if a.GetIdx() in highlights: #
                    atom_maps.append(a.GetAtomMapNum()) #
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = copy_edit_mol(new_mol).GetMol()
        new_mol = sanitize(new_mol) 
        highlight_clique = [] #
        for a in new_mol.GetAtoms(): #
            if a.GetAtomMapNum() in atom_maps: #
                highlight_clique.append(a.GetIdx()) #
        #if tmp_mol is not None: new_mol = tmp_mol 
        return new_mol, highlight_clique #
    except:
        return None,highlights


def get_assm_cands(mol, atoms, inter_label, cluster, inter_size):
    """
    mol : RDKit mol object
    atoms : list of atom indices from previous siblings of the current cluster
    inter_label : list of tuples of atom index and anchor_smiles of the inter cluster atoms
    cluster : list of atom indices in the parent cluster
    inter_size : number of atoms common between the current cluster and the parent cluster

    the new clique mol which is created by default has the atom map numbers set to its atom indices in the full molecule + 1. atom_map stores the correct atom map numbers for the atoms in the new clique mol to its atom indices in the full molecule by using the idxfunc function.


    """
    atoms = list(set(atoms))
    mol = get_clique_mol(mol, atoms)
    # if mol is None: # added for debugging. 
    #     return None
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

def get_inter_label(mol, atoms, inter_atoms):
    new_mol = get_clique_mol(mol, atoms)
    if new_mol.GetNumBonds() == 0: 
        inter_atom = list(inter_atoms)[0]
        for a in new_mol.GetAtoms():
            a.SetAtomMapNum(0)
        return new_mol, [ (inter_atom, Chem.MolToSmiles(new_mol)) ]

    inter_label = []
    for a in new_mol.GetAtoms():
        idx = idxfunc(a)
        if idx in inter_atoms and is_anchor(a, inter_atoms):
            inter_label.append( (idx, get_anchor_smiles(new_mol, idx)) )

    for a in new_mol.GetAtoms():
        a.SetAtomMapNum( 1 if idxfunc(a) in inter_atoms else 0 )
    return new_mol, inter_label

def is_anchor(atom, inter_atoms):
    for a in atom.GetNeighbors():
        if idxfunc(a) not in inter_atoms:
            return True
    return False

# new interlabel function for metals
def get_inter_label_metal(mol,atoms,inter_atoms,highlight_atoms_ligand):
    inter_label = []
    try:
        new_mol,highlight_new_mol=get_clique_mol_new(mol,atoms,highlight_atoms_ligand)

        if new_mol.GetNumBonds()==0:
            inter_atom=list(inter_atoms)[0]
            for a in new_mol.GetAtoms():
                a.SetAtomMapNum(0)
            # here we are changing back the atom map number for this clusters attached atoms to 2
            for a in new_mol.GetAtoms(): # 
                if a.GetIdx() in highlight_new_mol: #
                    a.SetAtomMapNum(2) #
            return new_mol, [(inter_atom, Chem.MolToSmiles(new_mol))]
        
        for a in new_mol.GetAtoms():
            idx = idxfunc(a)
            if idx in inter_atoms and is_anchor(a, inter_atoms):
                inter_label.append( (idx, get_anchor_smiles_metal(new_mol, idx, highlights=highlight_new_mol)) )

        for a in new_mol.GetAtoms():
            a.SetAtomMapNum( 1 if idxfunc(a) in inter_atoms else 0 )
        
        # here we are changing back the atom map number for this clusters attached atoms to metal centre to 2
        for a in new_mol.GetAtoms(): #
            if a.GetIdx() in highlight_new_mol: #
                a.SetAtomMapNum(2) #

        return new_mol, inter_label
    except:
        return None,inter_label
            
def get_anchor_smiles_metal(mol, anchor, metal_ponits, idxfunc=idxfunc):
    copy_mol = Chem.Mol(mol)
    for a in copy_mol.GetAtoms():
        idx = idxfunc(a)
        if idx in metal_ponits:
            a.SetAtomMapNum(2)
        elif idx == anchor:
            a.SetAtomMapNum(1)
        else:
            a.SetAtomMapNum(0)

    return get_smiles(copy_mol)

def get_anchor_smiles(mol, anchor, idxfunc=idxfunc):
    copy_mol = Chem.Mol(mol)
    for a in copy_mol.GetAtoms():
        idx = idxfunc(a)
        if idx == anchor: a.SetAtomMapNum(1)
        else: a.SetAtomMapNum(0)

    return get_smiles(copy_mol)

def recover_coordinates(n_atoms, edge_list, root_atom, export_xyz_path=None, atom_labels=None):
    """
    Reconstructs 3D coordinates from pairwise distances via optimization.

    n_atoms (int): Total number of atoms.
    edge_list (list of tuples): Each tuple is (i, j, distance).
    root_atom (int): Atom index to fix at origin (default: 0).
    export_xyz_path (str): Optional file path to save .xyz file.
    atom_labels (list of str): Optional list of element labels (e.g., ['Fe', 'N', 'N', 'Cl', ...]).

    Returns:
        coords (np.ndarray): Final coordinates of shape (n_atoms, 3).
    """

    dim = 3  # We’re reconstructing 3D coordinates
    x0 = np.random.rand(n_atoms, dim)
    x0[root_atom] = [0.0, 0.0, 0.0]  # Fix one atom at the origin to remove translational ambiguity
    x0 = x0.flatten()

    def loss(x):
        coords = x.reshape((n_atoms, dim))
        coords[root_atom] = [0.0, 0.0, 0.0]  
        error = 0.0
        for i, j, d_ij in edge_list:
            dist = np.linalg.norm(coords[i] - coords[j])
            error += (dist - d_ij) ** 2 
        return error

    def constraint_fixed_atom(x):
        return x[root_atom * 3: root_atom * 3 + 3]

    con = {'type': 'eq', 'fun': constraint_fixed_atom}

    res = minimize(loss, x0, constraints=[con], method='SLSQP', options={'maxiter': 1000})
    coords = res.x.reshape((n_atoms, dim))

    if export_xyz_path:
        if atom_labels:
            labels = atom_labels
        else: 
            raise ValueError("atom_labels must be provided if export_xyz_path is specified.")
        with open(export_xyz_path, 'w') as f:
            f.write(f"{n_atoms}\nGenerated by recover_coordinates\n")
            for label, (x, y, z) in zip(labels, coords):
                f.write(f"{label} {x:.5f} {y:.5f} {z:.5f}\n")
        print(f"Saved to {export_xyz_path}")
    else :
        raise ValueError("export_xyz_path must be provided to save the coordinates.")