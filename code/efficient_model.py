"""
梳理：
相较于死板且极大开销的inter_energy_between(双重循环)，vina_with_mol使用numpy库对
"""

import numpy as np
from repro_vina.simulation import Flex, Receptor, Ligand
import torch
from parameter import DEVICE, CURL_THRESHOLD, FLOAT_MAXIMUM, FLOAT_EPSILON


# 目前目标：对vina中的receptor_intra_energy和ligand_receptor_energy进行优化
def vina_with_mol(mol: Flex | Ligand, receptor: Receptor, weights: dict, ml_mode = False):
    if mol.size == 0:
        return 0.0
    rec_atoms = receptor.valid_atoms
    rec_vdws = [atom.vdw_radius for atom in rec_atoms]
    rec_coords = [atom.coordinate for atom in rec_atoms]
    rec_hs = [atom.is_hydrophobic() for atom in rec_atoms]
    rec_das = [int(atom.is_donor())*8+int(atom.is_acceptor())*4 for atom in rec_atoms]
    mol_atoms = [atom for atom in mol.valid_atoms if not atom.immobile]
    mol_vdws = [atom.vdw_radius for atom in mol_atoms]
    mol_coords = [atom.coordinate for atom in mol_atoms]
    h_mask = construct_hydrophobic_mask(rec_hs, [atom.is_hydrophobic() for atom in mol_atoms])
    mol_das = [int(atom.is_donor())*2+int(atom.is_acceptor()) for atom in mol_atoms]
    da_mask = construct_hydrogen_bonding_mask(rec_das, mol_das)
    dist_mat = construct_distance_matrix(rec_coords, mol_coords)
    opt_dist_mat = construct_optimal_distance_matrix(rec_vdws, mol_vdws)
    guass1 = np.sum(calculate_gauss_item(0, 0.5, 8, opt_dist_mat, dist_mat), axis=0)
    guass2 = np.sum(calculate_gauss_item(3, 2, 8, opt_dist_mat, dist_mat), axis=0)
    repulsion = np.sum(calculate_repulsion_item(0, 8, opt_dist_mat, dist_mat), axis=0)
    hydrophobic = np.sum(calculate_hardsigmod_item(0.5, 1.5, 8, opt_dist_mat, dist_mat, h_mask), axis=0)
    hydrogen_bonding = np.sum(calculate_hardsigmod_item(-0.7, 0, 8, opt_dist_mat, dist_mat, da_mask), axis=0)
    if ml_mode:
        guass1 = torch.from_numpy(guass1).to(DEVICE)
        guass2 = torch.from_numpy(guass2).to(DEVICE)
        repulsion = torch.from_numpy(repulsion).to(DEVICE)
        hydrophobic = torch.from_numpy(repulsion).to(DEVICE)
        hydrogen_bonding = torch.from_numpy(repulsion).to(DEVICE)
    energies = (
        weights["Gauss1"] * guass1
        + weights["Gauss2"] * guass2
        + weights["Repulsion"] * repulsion
        + weights["Hydrophobic"] * hydrophobic
        + weights["Hydrogen bonding"] * hydrogen_bonding)
    if ml_mode:
        return torch.sum(curl(energies, ml_mode=True))
    return np.sum(curl(energies))

def vina_with_mols(mols: list[Ligand], receptor: Receptor, weights: dict, ml_mode = False):
    rec_atoms = receptor.valid_atoms
    rec_vdws = [atom.vdw_radius for atom in rec_atoms]
    rec_coords = []
    for atom in rec_atoms:
        if isinstance(atom.coordinate, torch.Tensor):
            rec_coords.append(atom.coordinate.cpu())
        else:
            rec_coords.append(atom.coordinate)
    # rec_coords = [atom.coordinate for atom in rec_atoms]
    rec_hs = [atom.is_hydrophobic() for atom in rec_atoms]
    rec_das = [int(atom.is_donor())*8+int(atom.is_acceptor())*4 for atom in rec_atoms]
    curl_energies = torch.tensor(0.0) if ml_mode else 0.0
    for mol in mols:
        mol_atoms = [atom for atom in mol.valid_atoms if not atom.immobile]
        mol_vdws = [atom.vdw_radius for atom in mol_atoms]
        # mol_coords = [atom.coordinate for atom in mol_atoms]
        mol_coords = []
        for atom in mol_atoms:
            if isinstance(atom.coordinate, torch.Tensor):
                mol_coords.append(atom.coordinate.cpu())
            else:
                mol_coords.append(atom.coordinate)
        mol_hs = [atom.is_hydrophobic() for atom in mol_atoms]
        h_mask = construct_hydrophobic_mask(rec_hs, mol_hs)
        mol_das = [int(atom.is_donor())*2+int(atom.is_acceptor()) for atom in mol_atoms]
        da_mask = construct_hydrogen_bonding_mask(rec_das, mol_das)
        dist_mat = construct_distance_matrix(rec_coords, mol_coords)
        opt_dist_mat = construct_optimal_distance_matrix(rec_vdws, mol_vdws)
        guass1 = np.sum(calculate_gauss_item(0, 0.5, 8, opt_dist_mat, dist_mat), axis=0)
        guass2 = np.sum(calculate_gauss_item(3, 2, 8, opt_dist_mat, dist_mat), axis=0)
        repulsion = np.sum(calculate_repulsion_item(0, 8, opt_dist_mat, dist_mat), axis=0)
        hydrophobic = np.sum(calculate_hardsigmod_item(0.5, 1.5, 8, opt_dist_mat, dist_mat, h_mask), axis=0)
        hydrogen_bonding = np.sum(calculate_hardsigmod_item(-0.7, 0, 8, opt_dist_mat, dist_mat, da_mask), axis=0)
        if ml_mode:
            guass1 = torch.from_numpy(guass1).to(DEVICE)
            guass2 = torch.from_numpy(guass2).to(DEVICE)
            repulsion = torch.from_numpy(repulsion).to(DEVICE)
            hydrophobic = torch.from_numpy(hydrophobic).to(DEVICE)
            hydrogen_bonding = torch.from_numpy(hydrogen_bonding).to(DEVICE)
        energies = (
            weights["Gauss1"] * guass1
            + weights["Gauss2"] * guass2
            + weights["Repulsion"] * repulsion
            + weights["Hydrophobic"] * hydrophobic
            + weights["Hydrogen bonding"] * hydrogen_bonding
        )
        if ml_mode:
            curl_energies += torch.sum(curl(energies, ml_mode=True))
        else:
            curl_energies += np.sum(curl(energies))
    return curl_energies

def construct_distance_matrix(rec_coords, mol_coords):
    # input: rec_coords: [len(rec_atoms), 3], mol_coords: [len(mol_atoms), 3]
    # output: dist_mat: [len(rec_coords), len(mol_coords)]
    rec_arr = np.array([coord for coord in rec_coords])
    mol_arr = np.array([coord for coord in mol_coords])
    dist = rec_arr[:, np.newaxis, :] - mol_arr[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(dist**2, axis=2))
    return dist_mat

def construct_optimal_distance_matrix(rec_vdws, mol_vdws):
    # input: rec_vdws: [len(rec_atoms)], mol_vdws: [len(mol_atoms)]
    # output: opt_dist_mat: [len(rec_atoms), len(mol_atoms)]
    rec_arr = np.array(rec_vdws)
    mol_arr = np.array(mol_vdws)
    opt_dist_mat = rec_arr[:, np.newaxis] + mol_arr[np.newaxis, :]
    opt_dist_mat[rec_arr==0.0, :] = 0.0
    opt_dist_mat[:, mol_arr==0.0] = 0.0
    return opt_dist_mat

def construct_hydrophobic_mask(rec_hs, mol_hs):
    # input: rec_hbs: [len(rec_hbs)], mol_hbs: [len(mol_hbs)]
    # output: h_mat: [len(rec_hbs), len(mol_hbs)]
    return (np.array(rec_hs)[:, np.newaxis] & np.array(mol_hs)).astype(np.int8)

def construct_hydrogen_bonding_mask(rec_das, mol_das):
    # input: rec_ds: [len(rec_ds)], rec_as: [len(rec_as)], mol_ds: [len(mol_ds)], mol_as: [len(mol_as)]
    # output: hb_mat: [len(rec_hbs), len(mol_hbs)]
    mask = np.array(rec_das)[:, np.newaxis] | np.array(mol_das)
    da_mask = (mask&9 == 9) | (mask&6 == 6)
    mask[~da_mask] = 0
    mask[da_mask] = 1
    return mask

def calculate_gauss_item(offset, width, cutoff, opt_dist_mat, dist_mat):
    assert opt_dist_mat.shape == dist_mat.shape
    act_dist_mat = dist_mat - opt_dist_mat
    energy = np.exp(-((act_dist_mat - offset) / width)**2)
    energy[dist_mat >= cutoff] = 0.0
    return energy

def calculate_repulsion_item(boundary, cutoff: float, opt_dist_mat, dist_mat):
    assert opt_dist_mat.shape == dist_mat.shape
    act_dist_mat = dist_mat - opt_dist_mat
    energy = act_dist_mat ** 2
    energy[dist_mat >= cutoff] = 0.0
    energy[act_dist_mat > boundary] = 0.0
    return energy

def calculate_hardsigmod_item(lower_bound, upper_bound, cutoff, opt_dist_mat, dist_mat, mask):
    assert opt_dist_mat.shape == dist_mat.shape
    assert dist_mat.shape == mask.shape
    act_dist_mat = dist_mat - opt_dist_mat
    transition_mask = (act_dist_mat >= lower_bound) & (act_dist_mat <= upper_bound)
    energy = np.zeros_like(dist_mat)
    energy[act_dist_mat < lower_bound] = 1
    energy[transition_mask] = 1 - (act_dist_mat[transition_mask] - lower_bound) / (upper_bound - lower_bound)
    energy *= mask
    energy[dist_mat >= cutoff] = 0.0
    return energy

# 当计算的能量大于0时，通过平滑变换将能量限制在阈值以内
def curl(energy, threshold: float = CURL_THRESHOLD, ml_mode=False):
    if threshold < 0.1 * FLOAT_MAXIMUM:
        tmp = 0 if threshold < FLOAT_EPSILON else (threshold / (threshold+energy))
        if ml_mode:
            return torch.where(energy > 0, energy*tmp, energy)
        return np.where(energy > 0, energy*tmp, energy)
    return energy
