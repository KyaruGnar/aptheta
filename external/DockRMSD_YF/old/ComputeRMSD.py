from rdkit import Chem
from rdkit.Chem import AllChem
import os,re
import numpy as np
from isoRMSD import GetBestRMSD
from tqdm import tqdm
import argparse
from collections import Counter
import rdkit

def show_rmsd(rmsdList, thrs=[0.5,1.0,2.0,3.0,5.0]):
    print(f'{len(rmsdList[rmsdList>0])} items in total...')
    print('Avg RMSD:', np.mean(rmsdList[rmsdList>0]))
    for thr in thrs:
        print(f'ratio of rmsd<{thr}: {np.mean(rmsdList[rmsdList>0]<thr):.3f}')

from scipy.optimize import linear_sum_assignment
def hungarianRMSD(mol1, mol2):
    at1 = np.array([i.GetSymbol() for i in mol1.GetAtoms()])
    at2 = np.array([i.GetSymbol() for i in mol2.GetAtoms()])
    isSame = at1[:,None]==at2[None]
    xyz1 = mol1.GetConformers()[0].GetPositions()
    xyz2 = mol2.GetConformers()[0].GetPositions()
    cost = np.sum((xyz1[:,None] - xyz2[None])**2, axis=-1)
    cost[~isSame] = np.inf
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(cost[row_ind,col_ind].mean()),xyz1[row_ind],xyz2[col_ind]
def RodriguesMatrixModel(src, dst, scale=None):
    # 计算比例关系
    if scale is None:
        scale = np.sum(np.sqrt(np.sum((dst - np.mean(dst, axis=0)) ** 2, axis=1))) / np.sum(np.sqrt(np.sum((src - np.mean(src, axis=0)) ** 2, axis=1)))
    # 计算旋转矩阵
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src = src - src_mean
    dst = dst - dst_mean
    H = np.dot(src.T, dst)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    # 计算平移矩阵
    t = dst_mean.T - scale * np.dot(R, src_mean.T)
    return R, t, scale
def PointBasedBestRMSD(mol1, mol2):
    rmsd,xyz1,xyz2 = hungarianRMSD(mol1,mol2)
    R,t,_ = RodriguesMatrixModel(xyz1,xyz2)
    aligned_xyz1 = xyz1@R.T+t
    alignedrmsd = np.sqrt(np.mean(np.sum((aligned_xyz1-xyz2)**2, axis=-1)))
    return alignedrmsd,rmsd,None

# blind UniMolLM
# python .\CompuateRMSD.py --input "../unimollm_predictions/blind_predicted_ligand_pose_(.*?).mol" --ref "../unimollm_predictions/true_ligand_pose_*.mol"
# pocket UniMolLM
# python .\CompuateRMSD.py --input "../unimollm_predictions/pocket_predicted_ligand_pose_(.*?).mol" --ref "../unimollm_predictions/true_ligand_pose_*.mol"
# blind vina
# python .\CompuateRMSD.py --input "./vina_output/(.*?)_ligand_protein_predicted.mol" --ref "./casf2016/*_ligand.mol" --align None
# pocket vina
# python .\CompuateRMSD.py --input "./vina_output/(.*?)_ligand_pocket_predicted.mol" --ref "./casf2016/*_ligand.mol" --align None

## HM_res
# blind fpocket_vina(default)
# python .\CompuateRMSD.py --input "./hm_docking/fpocket_vina_res_mol_by_hm/vina_(.*?)_protein_ligand_0_out.mol" --ref "./casf2016/*_ligand.mol" --align SubStrMatch

# blind fpocket_vina: d1_mol
# python .\CompuateRMSD.py --input "./hm_docking/d1_mol/vina_(.*?)_protein_ligand_0_out.mol" --ref "./casf2016/*_ligand.mol" --align SubStrMatch

# blind fpocket_vina: d2_mol
# python .\CompuateRMSD.py --input "./hm_docking/d2_mol/vina_(.*?)_protein_ligand_0_out.mol" --ref "./casf2016/*_ligand.mol" --align SubStrMatch

# blind fpocket_vina: d3_mol
# python .\CompuateRMSD.py --input "./hm_docking/d3_mol/vina_(.*?)_protein_ligand_0_out.mol" --ref "./casf2016/*_ligand.mol" --align SubStrMatch

# blind fpocket_vina: size_8_mol
# python .\CompuateRMSD.py --input "./hm_docking/size_8_mol/vina_(.*?)_protein_ligand_0_out.mol" --ref "./casf2016/*_ligand.mol" --align SubStrMatch

# pocket uni dock
# python .\CompuateRMSD.py --input "./hm_docking/result_caft/(.*?)_protein_ligand_out.mol" --ref "./casf2016/*_ligand.mol" --align SubStrMatch

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='.\\test.mol2')
parser.add_argument('--ref', default='.\\2c0k_C_HEM.mol2')
parser.add_argument('--align', default='Auto', choices=['None','SubStrMatch','HunMatch','Auto'], type=str)
args = parser.parse_args()

if __name__=='__main__':
    inputDir,inputPattern = os.path.split(args.input)
    print(inputDir, inputPattern)
    refDir,refPattern = os.path.split(args.ref)

    nameList = []
    for file in os.listdir(inputDir):
        if os.path.isdir(file): continue

        name = re.findall(inputPattern, file)
        if len(name)!=1: continue
        nameList.append([file,name[0]])
    
    rmsdList,alignedrmsdList = [],[]
    logList = []
    for file,name in tqdm(nameList):
        if args.align=='SubStrMatch':
            try:
                mol1 = Chem.RemoveAllHs(Chem.MolFromMolFile(os.path.join(inputDir,file)))
                mol2 = Chem.RemoveAllHs(Chem.MolFromMolFile(args.ref.replace('*',name)))
                alignedrmsd,rmsd,_ = GetBestRMSD(mol1,mol2)
                log = 's1_SubStrMatch'
            except:
                alignedrmsd,rmsd = -1,-1
                log = f'error_in_{name}'
        elif args.align=='Auto':
            try:
                mol1 = Chem.RemoveAllHs(Chem.MolFromMol2File(os.path.join(inputDir,file)))
                mol2 = Chem.RemoveAllHs(Chem.MolFromMol2File(args.ref.replace('*',name)))
                try:
                    alignedrmsd,rmsd,_ = GetBestRMSD(mol1,mol2)
                    log = 's1_SubStrMatch'
                except:
                    alignedrmsd,rmsd,_ = PointBasedBestRMSD(mol1,mol2)
                    log = 's1_HunMatch'
            except:
                try:
                    mol1 = Chem.RemoveAllHs(Chem.MolFromMol2File(os.path.join(inputDir,file),sanitize=False),sanitize=False)
                    mol2 = Chem.RemoveAllHs(Chem.MolFromMol2File(args.ref.replace('*',name),sanitize=False),sanitize=False)
                    alignedrmsd,rmsd,_ = PointBasedBestRMSD(mol1,mol2)
                    log = 's0_HunMatch'
                except:
                    alignedrmsd,rmsd = -1,-1
                    log = f'error_in_{name}'
        elif args.align=='None':
            try:
                mol1 = Chem.RemoveAllHs(Chem.MolFromMol2File(os.path.join(inputDir,file),sanitize=False),sanitize=False)
                xyz1 = mol1.GetConformers()[0].GetPositions()
                mol2 = Chem.RemoveAllHs(Chem.MolFromMol2File(args.ref.replace('*',name),sanitize=False),sanitize=False)
                xyz2 = mol2.GetConformers()[0].GetPositions()

                a1 = "".join([i.GetSymbol() for i in mol1.GetAtoms()])
                a2 = "".join([i.GetSymbol() for i in mol2.GetAtoms()])
                assert a1==a2

                R,t,_ = RodriguesMatrixModel(xyz1,xyz2)
                aligned_xyz1 = xyz1@R.T+t

                rmsd = np.sqrt(np.mean(np.sum((xyz1-xyz2)**2, axis=-1)))
                alignedrmsd = np.sqrt(np.mean(np.sum((aligned_xyz1-xyz2)**2, axis=-1)))
                log = 's0_None'
            except:
                alignedrmsd,rmsd = -1,-1
                log = f'error_in_{name}'
        elif args.align=='HunMatch':
            try:
                mol1 = Chem.RemoveAllHs(Chem.MolFromMolFile(os.path.join(inputDir,file),sanitize=False),sanitize=False)
                mol2 = Chem.RemoveAllHs(Chem.MolFromMolFile(args.ref.replace('*',name),sanitize=False),sanitize=False)
                alignedrmsd,rmsd,_ = PointBasedBestRMSD(mol1,mol2)
                log = 's0_HunMatch'
            except:
                alignedrmsd,rmsd = -1,-1
                log = f'error_in_{name}'

        rmsdList.append(rmsd)
        alignedrmsdList.append(alignedrmsd)
        logList.append(log)

    print('LOG INFO:')
    print(Counter(logList).most_common(len(logList)))
    print('==========RMSD==========')
    show_rmsd(np.array(rmsdList))
    print('==========AlignedRMSD==========')
    show_rmsd(np.array(alignedrmsdList))


