import itertools,glob
from scipy.optimize import linear_sum_assignment
import Bio.PDB as PDB
import numpy as np
from Bio.PDB import PDBIO
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.Seq import Seq
from Bio.SeqUtils import seq1 as seq3_1
import rdkit
from rdkit import Chem
from collections import Counter
# pairwise2有弃用风险故使用新函数
# from Bio import pairwise2
from Bio.Align import PairwiseAligner

def flatten_list(list_1):
    """
    将嵌套列表展开为一维列表
    :param list_1: 嵌套列表
    :return: 一维列表
    """
    result = []
    for item in list_1:
        if isinstance(item, list) or isinstance(item, tuple):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def sorted_permulate(seqList_sorted):
    if len(seqList_sorted)==1:
        return [[0]]
    permutationList = []
    permutate = []
    for i in range(len(seqList_sorted)):
        permutate.append(i)

        if i==0: continue
        
        if seqList_sorted[i]!=seqList_sorted[i-1]:
            permutationList.append( itertools.permutations(permutate[:-1]) )
            permutate = permutate[-1:]
            if i==len(seqList_sorted)-1:
                permutationList.append( permutate )
        elif i==len(seqList_sorted)-1:
            permutationList.append( itertools.permutations(permutate) )

    if len(permutationList)>0:
        valid_permutation = permutationList[0]
        for i in range(1,len(permutationList)):
            valid_permutation = itertools.product(valid_permutation, permutationList[i])

        return valid_permutation
    else:
        return [()]

pippo = set(['A','C','G','U','I','T'])
def extract_protein_sequence(pdb_file_path: str):
    if pdb_file_path is not None and pdb_file_path.endswith('.pdb'):
        # 创建PDB解析器
        parser = PDB.PDBParser(QUIET=True)
        # 读取PDB文件
        structure = parser.get_structure('protein', pdb_file_path)
    elif pdb_file_path.endswith('cif'):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file_path)
    else:
        print(f'fucking unsupported file format in {pdb_file_path}!')
        exit(0)

    # 初始化蛋白质序列
    protein_sequence = ''

    # 遍历模型、链和残基
    if len(structure)>1:
        print('found more than 1 model, using the first one...')
    chainList,xyzList = [],[]
    print(len(structure), 'model found!')
    for model in structure:
        for chain in model:
            seq = ""
            xyz = []
            for residue in chain:
                # 检查残基是否为氨基酸
                if PDB.is_aa(residue):
                    # 获取氨基酸的单字母代码并追加到蛋白质序列中
                    aa = seq3_1(residue.get_resname())
                    for atom in residue.get_atoms():
                        if atom.get_name()=='CA':
                            seq += aa
                            xyz.append(atom.get_coord())
                            break
                elif residue.get_resname() in pippo: # DNA/RNA
                    # 获取核酸的单字母代码并追加到RNA序列中
                    seq += residue.get_resname()
                    for atom in residue.get_atoms():
                        if atom.get_name().strip()=='P':
                            xyz.append(atom.get_coord())
                            break
            if len(seq)>0: 
                assert len(seq)==len(xyz)
                chainList.append(seq.upper())
                xyzList.append(np.array(xyz,dtype=np.float32))
        break
    
    return {'seq':chainList, 'xyz':xyzList}

# 为兼容pairwise2.align.globalxx的废弃风险接口, 以PairwiseAligner功能编写的替代品
def globalxx(seqi: str, seqj: str):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0
    alignments = aligner.align(seqi, seqj)
    best = alignments[0]
    seqA_idx = best.aligned[0]  # seqi 的匹配区间列表
    seqB_idx = best.aligned[1]  # seqj 的匹配区间列表
    aligned_i = []
    aligned_j = []
    i_idx = j_idx = 0
    for (start_i, end_i), (start_j, end_j) in zip(seqA_idx, seqB_idx):
        # gap in seqi
        while i_idx < start_i:
            aligned_i.append(seqi[i_idx])
            aligned_j.append('-')
            i_idx += 1
        # gap in seqj
        while j_idx < start_j:
            aligned_i.append('-')
            aligned_j.append(seqj[j_idx])
            j_idx += 1
        # match segment
        aligned_i.extend(seqi[start_i:end_i])
        aligned_j.extend(seqj[start_j:end_j])
        i_idx = end_i
        j_idx = end_j
    # 补齐末尾
    while i_idx < len(seqi):
        aligned_i.append(seqi[i_idx])
        aligned_j.append('-')
        i_idx += 1
    while j_idx < len(seqj):
        aligned_i.append('-')
        aligned_j.append(seqj[j_idx])
        j_idx += 1
    seqA_aligned = ''.join(aligned_i)
    seqB_aligned = ''.join(aligned_j)
    score = best.score
    return seqA_aligned, seqB_aligned, score


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ref_rec_path') # xxx/xxx.pdb
parser.add_argument('--ref_lig_path') # xxx/*.sdf or xxx/xxx.sdf
parser.add_argument('--pre_rec_path') # xxx/xxx.pdb
parser.add_argument('--pre_lig_path') # xxx/*.sdf or xxx/xxx.sdf
parser.add_argument('--atom_identity_rule', default='an', choices=['an', 'a'], type=str)

args = parser.parse_args()

if __name__=='__main__':
    #START
    # # M = Chem.MolFromMolFile("./data/3tbe/ligands/ETM_C_422_adapted.sdf")
    # M = Chem.MolFromMolFile("./data/3tbe/mlxform/out0_ligand1_5.sdf", sanitize=False)
    # args.ref_rec_path = './data/7zx3/protein/7zx3_atoms.pdb'
    # args.pre_rec_path = './data/7zx3/protein/7zx3_atoms.pdb'
    # args.ref_lig_path = './data/7zx3/ligands/SF4_X_701_adapted.sdf'
    # args.pre_lig_path = './data/7zx3/mlxform/out0_ligand1_1.sdf'
    # args.atom_identity_rule = 'a'
    #END
    prot_ref = extract_protein_sequence(args.ref_rec_path)
    # print(len(prot_ref['seq'][0]),prot_ref['xyz'][0].shape)
    if '*' in args.ref_lig_path:
        ligs_ref = [Chem.MolFromMolFile(i, sanitize=False) for i in glob.glob(args.ref_lig_path)]
    else:
        ligs_ref = Chem.rdmolops.GetMolFrags(Chem.MolFromMolFile(args.ref_lig_path, sanitize=False), asMols=True, sanitizeFrags=False)

    prot_pre = extract_protein_sequence(args.pre_rec_path)
    if '*' in args.pre_lig_path:
        ligs_pre = [Chem.MolFromMolFile(i, sanitize=False) for i in glob.glob(args.pre_lig_path)]
    else:
        ligs_pre = Chem.rdmolops.GetMolFrags(Chem.MolFromMolFile(args.pre_lig_path, sanitize=False), asMols=True, sanitizeFrags=False)

    print('====================RMSD-ligand COMPUTING PROGRAM====================')
    print(f'{len(prot_ref["seq"])} chains in ref protein, {len(ligs_ref)} ligands in ref lignds...')
    print(f'{len(prot_pre["seq"])} chains in pre protein, {len(ligs_pre)} ligands in pre lignds...')
    assert len(prot_ref["seq"])==len(prot_pre["seq"])
    assert len(ligs_ref)==len(ligs_pre)

    # warning
    if (np.sort(prot_ref['seq'])!=np.sort(prot_pre['seq'])).any():
        print('warning!!! different length for the ref pdb and pre pdb!!! try to solve this...')
        
        # find chain mapping by sequence alignment
        costMat = np.ones((len(prot_ref['seq']),len(prot_ref['seq'])), dtype=np.float32)

        for i in range(len(prot_ref['seq'])):
            for j in range(len(prot_pre['seq'])):
                seqi,seqj = prot_ref['seq'][i],prot_pre['seq'][j]

                l = min(len(prot_ref['seq'][i]), len(prot_pre['seq'][j]))
                
                # costMat[i,j] = 1 - pairwise2.align.globalxx(Seq(seqi) , Seq(seqj) )[0].score / l
                costMat[i,j] = 1 - globalxx(Seq(seqi) , Seq(seqj))[2] / l

        row_ind, col_ind = linear_sum_assignment(costMat)

        prot_ref['seq'],prot_ref['xyz'] = [prot_ref['seq'][i] for i in row_ind],[prot_ref['xyz'][i] for i in row_ind]
        prot_pre['seq'],prot_pre['xyz'] = [prot_pre['seq'][i] for i in col_ind],[prot_pre['xyz'][i] for i in col_ind]

        # unify the chain length for reference sequences and prediction sequences
        for i in range(len(prot_ref['seq'])):
            seqi,seqj = prot_ref['seq'][i],prot_pre['seq'][i]

            if len(seqj)<len(seqi):
                print('Warning!!! length of predicted sequence is shorter than the reference sequence...')

            # alignments = pairwise2.align.globalxx(seqi, seqj)[0]
            seqA, seqB, _ = globalxx(seqi, seqj)
            isUsed1,isUsed2 = [],[]
            # for t1,t2 in zip(alignments.seqA, alignments.seqB):
            for t1,t2 in zip(seqA, seqB):
                if t1=='-':
                    isUsed2.append(False)
                elif t2=='-':
                    isUsed1.append(False)
                elif t1==t2:
                    isUsed1.append(True)
                    isUsed2.append(True)
                else:
                    print('Fuck something wrong!!!')
                    exit(0)

            assert len(isUsed1)==len(prot_ref['seq'][i])
            assert len(isUsed2)==len(prot_pre['seq'][i])
            
            prot_ref['seq'][i] = "".join([a for j,a in zip(isUsed1,prot_ref['seq'][i]) if j])
            prot_pre['seq'][i] = "".join([a for j,a in zip(isUsed2,prot_pre['seq'][i]) if j])
            prot_ref['xyz'][i] = np.array([a for j,a in zip(isUsed1,prot_ref['xyz'][i]) if j], dtype=np.float32)
            prot_pre['xyz'][i] = np.array([a for j,a in zip(isUsed2,prot_pre['xyz'][i]) if j], dtype=np.float32)
    
        print(prot_ref['seq'], [len(i) for i in prot_ref['seq']])
        print(prot_pre['seq'], [len(i) for i in prot_pre['seq']])
    
    # sort all seq by length and char
    ligs_ref_seq = ["".join(sorted([a.GetSymbol() for a in i.GetAtoms()])) for i in ligs_ref]
    ligs_pre_seq = ["".join(sorted([a.GetSymbol() for a in i.GetAtoms()])) for i in ligs_pre]

    prot_ref_sortedIdx,prot_pre_sortedIdx = np.argsort(prot_ref['seq']),np.argsort(prot_pre['seq'])
    ligs_ref_sortedIdx,ligs_pre_sortedIdx = np.argsort(ligs_ref_seq),np.argsort(ligs_pre_seq)

    prot_ref['seq'],prot_pre['seq'] = [prot_ref['seq'][i] for i in prot_ref_sortedIdx],[prot_pre['seq'][i] for i in prot_pre_sortedIdx]
    prot_ref['xyz'],prot_pre['xyz'] = [prot_ref['xyz'][i] for i in prot_ref_sortedIdx],[prot_pre['xyz'][i] for i in prot_pre_sortedIdx]
    ligs_ref,ligs_pre = [ligs_ref[i] for i in ligs_ref_sortedIdx],[ligs_pre[i] for i in ligs_pre_sortedIdx]
    ligs_ref_seq,ligs_pre_seq = [ligs_ref_seq[i] for i in ligs_ref_sortedIdx],[ligs_pre_seq[i] for i in ligs_pre_sortedIdx]

    if (np.sort(ligs_ref_seq)!=np.sort(ligs_pre_seq)).any():
        print('ref_lig:', np.sort(ligs_ref_seq))
        print('pre_lig:', np.sort(ligs_pre_seq))
        print('ligands not match... exit!')

    # compute some array to be used for reference data

    aa_seq_ref = "".join(prot_ref['seq'])
    aa_xyz_ref = np.vstack(prot_ref['xyz'])

    if args.atom_identity_rule=='an':
        at_seq_ref = [[j.GetSymbol()+'-'+str(sorted([k.GetSymbol() for k in j.GetNeighbors()])) for j in i.GetAtoms()] for i in ligs_ref]
    else:
        at_seq_ref = [[j.GetSymbol() for j in i.GetAtoms()] for i in ligs_ref]
        
    at_xyz_ref = [i.GetConformers()[0].GetPositions() for i in ligs_ref]

    str_mol_ref = "\n".join([";".join(sorted(i)) for i in at_seq_ref])

    print('computing RMSD-ligand by enumerating all chain mapping and ligand mapping...')

    best_RMSD = np.inf
    # for chain_map in itertools.permutations(range(len(prot_pre['seq']))):
    for chain_map in sorted_permulate(prot_pre['seq']):
        chain_map = flatten_list(chain_map)

        aa_seq_pre = "".join([prot_pre['seq'][i] for i in chain_map])
        aa_xyz_pre = np.vstack([prot_pre['xyz'][i] for i in chain_map])

        if aa_seq_ref!=aa_seq_pre: continue

        # for lig_map in itertools.permutations(range(len(ligs_pre))):
        for lig_map in sorted_permulate(ligs_pre_seq):
            lig_map = flatten_list(lig_map)

            ligs_pre_ = [ligs_pre[i] for i in lig_map]
            if args.atom_identity_rule=='an':
                at_seq_pre = [[j.GetSymbol()+'-'+str(sorted([k.GetSymbol() for k in j.GetNeighbors()])) for j in i.GetAtoms()] for i in ligs_pre_]
            else:
                at_seq_pre = [[j.GetSymbol() for j in i.GetAtoms()] for i in ligs_pre_]
            at_xyz_pre = [i.GetConformers()[0].GetPositions() for i in ligs_pre_] # N, atn,3
            str_mol_pre = "\n".join([";".join(sorted(i)) for i in at_seq_pre])

            if str_mol_ref!=str_mol_pre: continue

            rmsd = 0
            for seq_pre,seq_ref,xyz_pre,xyz_ref in zip(at_seq_pre,at_seq_ref,at_xyz_pre, at_xyz_ref):

                costMat = np.sqrt(np.sum((xyz_pre[:,None] - xyz_ref[None])**2, axis=-1)+1e-8) # atn,atn
                isValid = np.array(seq_pre)[:,None]==np.array(seq_ref)[None]
                costMat[~isValid] = np.inf

                row_ind, col_ind = linear_sum_assignment(costMat)

                xyz_pre,xyz_ref = xyz_pre.copy()[row_ind],xyz_ref.copy()[col_ind] # atn,3

                rmsd += np.sqrt(np.mean(np.sum((xyz_pre-xyz_ref)**2, axis=-1)))
                
            rmsd /= len(at_seq_pre)

            if rmsd < best_RMSD:
                best_RMSD = rmsd
    if best_RMSD==np.inf:
        print('no ligand match found. try to set --atom_identity_rule to "a" for atom only...')
    else:
        print('done!')
        print('====================RESULT====================')
        print(f'RMSD-lig = {best_RMSD:.5f}')

# python ./pylib/CompuateRMSDlig.py --ref_rec_path ./datasets/AF3/reference/target_new_modeling/8CQA/8cqa_rec.pdb --ref_lig_path "./datasets/AF3/reference/target_new_modeling/8CQA/8cqa_*.sdf" --pre_rec_path ./datasets/AF3/prediction/rec_lig/fold_8cqa_rec_lig/fold_8cqa_rec_lig_model_0_rec.pdb --pre_lig_path "./datasets/AF3/prediction/rec_lig/fold_8cqa_rec_lig/fold_8cqa_rec_lig_model_0_*_lig.sdf"

# python ./pylib/CompuateRMSDlig.py --ref_rec_path ./datasets/AF3/reference/target_new_modeling/8CQA/8cqa_rec.pdb --ref_lig_path "./datasets/AF3/reference/target_new_modeling/8CQA/8cqa_*.sdf" --pre_rec_path ./datasets/AF3/reference/target_new_modeling/8CQA/8cqa_rec.pdb --pre_lig_path ./output/PDB/vina_multi_lig/8CQA/ligs_docked.sdf