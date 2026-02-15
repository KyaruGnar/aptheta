"""
dock_score.py文件为临时文件, 合并了原先的energy.py和main_model.py,并删除了预计算功能
原实现功能完整版需使用repro_vina包

Last update: 2026-02-15 by Junlin_409
version: 1.0.0
"""

# 引入区
import math
from itertools import combinations, product
from parameter import VINA_WEIGHTS, CURL_THRESHOLD, FLOAT_MAXIMUM, FLOAT_EPSILON
from repro_vina.utility import HardSigmod
from repro_vina.simulation import GeometricConstraint, Atom, AtomPair, RigidMolecule, Molecule, Flex, Receptor, Ligand, SimpleMolecule
from efficient_model import vina_with_mol, vina_with_mols

"""
原energy.py文件部分, 旨在归纳分子对接时的能量计算:
(1)势能类: 实现了相关势能项的计算
(2)计算器: 实现原子对(集), 分子内, 分子间的计算, 并设置相关势能项来完成模型的能量计算
(3)预计算: 根据原子类型实现计算单元并进行预计算, 后续根据输入获取计算结果
"""

# 势能基类
class Potential:
    def calculate(self, pair: AtomPair, r: float) -> float: # pylint: disable=W0613
        return 0.0


# 高斯项计算类
class Gauss(Potential):
    def __init__(self, offset: float, width: float, cutoff: float) -> None:
        self.offset = offset
        self.width = width
        self.cutoff = cutoff

    def calculate(self, pair: AtomPair, r: float) -> float:
        if r >= self.cutoff:
            return 0.0
        d = r - pair.optimal_distance()
        return math.exp(-math.pow((d-self.offset)/self.width, 2))


# 斥力项计算类
class Repulsion(Potential):
    def __init__(self, boundary: float, cutoff: float) -> None:
        self.boundary = boundary
        self.cutoff = cutoff

    def calculate(self, pair: AtomPair, r: float) -> float:
        if r >= self.cutoff:
            return 0.0
        d = r - pair.optimal_distance()
        if d > self.boundary:
            return 0.0
        return math.pow(d, 2)


# 疏水项计算类
class Hydrophobic(Potential):
    def __init__(self, lower_bound: float, upper_bound: float, cutoff: float) -> None:
        self.hard_sigmod = HardSigmod(-upper_bound, -lower_bound)
        self.cutoff = cutoff

    def calculate(self, pair: AtomPair, r: float) -> float:
        if r >= self.cutoff:
            return 0.0
        d = r - pair.optimal_distance()
        if pair.atoms[0].is_hydrophobic() and pair.atoms[1].is_hydrophobic():
            return self.hard_sigmod.calculate(-d)
        return 0.0


# 氢键项计算类
class HydrogenBonding(Potential):
    def __init__(self, lower_bound: float, upper_bound: float, cutoff: float) -> None:
        self.hard_sigmod = HardSigmod(-upper_bound, -lower_bound)
        self.cutoff = cutoff

    def calculate(self, pair: AtomPair, r: float) -> float:
        if r >= self.cutoff:
            return 0.0
        d = r - pair.optimal_distance()
        if pair.is_donor_acceptor():
            return self.hard_sigmod.calculate(-d)
        return 0.0


# 线性吸引项计算类
class LinearAttraction(Potential):
    def __init__(self, cutoff: float) -> None:
        self.cutoff = cutoff

    def calculate(self, pair: AtomPair, r: float) -> float:
        if r >= self.cutoff:
            return 0.0
        if pair.is_gluey():
            return r
        return 0.0


# 当计算的能量大于0时，通过平滑变换将能量限制在阈值以内
def curl(enery: float, threshold: float = CURL_THRESHOLD) -> float:
    if enery > 0 and threshold < 0.1 * FLOAT_MAXIMUM:
        tmp = 0 if threshold < FLOAT_EPSILON else (threshold / (threshold+enery))
        enery *= tmp
    return enery


# 能量运算类
class Calculator:
    def __init__(self, weights: dict | None = None) -> None:
        self.calculators: dict[str, Potential] = {}
        self.max_cutoff = 0.0
        self.calculators = {
            "Gauss1": Gauss(0, 0.5, 8),
            "Gauss2": Gauss(3, 2, 8),
            "Repulsion": Repulsion(0, 8),
            "Hydrophobic": Hydrophobic(0.5, 1.5, 8),
            "Hydrogen bonding": HydrogenBonding(-0.7, 0, 8),
            "Glue": LinearAttraction(20.0)
        }
        if weights is None:
            self.weights = VINA_WEIGHTS
        else:
            self.weights = weights
        self.max_cutoff = 20.0

    # 原子对能量计算
    def paired_energy_of(self, pair: AtomPair, r: float | None = None) -> float:
        energy = 0.0
        if r is None:
            r = pair.distance()
        for (term_name, term_class) in self.calculators.items():
            energy += self.weights[term_name] * term_class.calculate(pair, r)
        return energy

    # 原子对集能量计算
    def pairwise_energy_of(self, pairs: list[AtomPair]) -> float:
        return sum(curl(self.paired_energy_of(pair)) for pair in pairs)

    # 分子内部能量计算
    def intra_energy_of(self, molecule: Molecule) -> float:
        return self.pairwise_energy_of(molecule.pairs)

    # 分子间能量计算(使用原子类型xscore,且不考虑未初始化的原子,分子i为可变分子但不考虑其中无法移动的原子,分子j无限制)
    def inter_energy_between(self, molecule_i: Molecule, molecule_j: RigidMolecule):
        if not isinstance(molecule_i, Molecule):
            raise TypeError("Calculator_wrong_molecule_type_input")
        energy = 0.0
        for atom_i in molecule_i.valid_atoms:
            if atom_i.immobile:
                continue
            energy += curl(sum(self.paired_energy_of(AtomPair(atom_i, atom_j)) for atom_j in molecule_j.valid_atoms))
        return energy

    # 一个与旋转自由度有关的单调递增函数,使打分结果间的差异更明显
    def g(self, energy, torsion):
        return energy / (1 + self.weights["Rot"]*torsion)



"""
原main_model.py文件部分，旨在归纳对接模型:
(1)对接模型类: 基本的对接输入和分数计算
(2)多重检验类: 将output文件中的对接结果重新打分,与后续机器学习权重相关
"""

# 对接模型类
class DockingModel:
    def __init__(self, weights: dict | None = None) -> None:
        self.receptor = Receptor()
        self.flex = Flex()
        self.ligands: list[Ligand] = []
        self.flexible_atoms: list[Atom] = []
        self.pairs: dict[str, list[AtomPair]] = {"gluey": [], "inter": [], "other": []}
        self.calculator = Calculator(weights)

    # 设置受体(可能含有柔性部分)
    def set_receptor(self, receptor: Receptor, flex: Flex) -> None:
        self.flex = flex
        flex.activate()
        self.flexible_pairing(flex)
        self.receptor = receptor
        receptor.activate(flex)
        self.initialize_pairs(flex)

    # 设置配体
    def set_ligands(self, ligands: list[Ligand]) -> None:
        for ligand in ligands:
            ligand.activate()
            self.initialize_pairs(ligand)
            self.flexible_pairing(ligand)
            self.ligands.append(ligand)

    # 柔性分子添加
    def flexible_pairing(self, molecule: Molecule) -> None:
        for atom_i, atom_j in product(self.flexible_atoms, molecule.valid_atoms):
            if atom_j.immobile:
                continue
            pair = AtomPair(atom_i, atom_j)
            if pair.is_closure_clash() or pair.is_unmatched_closure_dummy():
                continue
            if pair.is_gluey():
                self.pairs["gluey"].append(pair)
            elif isinstance(atom_i.molecule, Ligand) or isinstance(atom_j.molecule, Ligand):
                self.pairs["inter"].append(pair)
            else:
                self.pairs["other"].append(pair)
        self.flexible_atoms.extend(atom for atom in molecule.valid_atoms if not atom.immobile)

    # 初始化对
    def initialize_pairs(self, molecule: Molecule | None) -> None:
        if molecule is None:
            return
        atom_neighbors: dict[Atom, set[Atom]] = {}
        for atom in molecule.valid_atoms:
            atom_neighbors[atom] = atom.neighbors_within(3)
        for atom_i, atom_j in combinations(molecule.valid_atoms, 2):
            if atom_j in atom_neighbors[atom_i]:
                continue
            pair = AtomPair(atom_i ,atom_j)
            if pair.geometric_relation(molecule.atom_positioner) == GeometricConstraint.VARIABLE:
                if pair.is_closure_clash() or pair.is_unmatched_closure_dummy():
                    continue
                if pair.is_gluey():
                    self.pairs["gluey"].append(pair)
                elif isinstance(molecule, Ligand):
                    molecule.pairs.append(pair)
                else:
                    self.pairs["other"].append(pair)

    def validate(self) -> None:
        if self.receptor.size == 0 or len(self.ligands) == 0:
            raise TypeError("DockingModel_loss_of_structure")

    # vina分数计算
    def vina(self, best_intra_score=0.0, ml_mode = False) -> tuple[float, float, float]:
        self.validate()
        # flex - receptor
        receptor_intra_energy = vina_with_mol(self.flex, self.receptor, self.calculator.weights, ml_mode)
        # flex_i - flex_i and flex_i - flex_j
        flex_intra_energy = self.calculator.pairwise_energy_of(self.pairs["other"])
        # ligand_i - ligand_i / ligand - receptor
        ligand_receptor_energy = vina_with_mols(self.ligands, self.receptor, self.calculator.weights, ml_mode)
        ligands_intra_energy = 0.0
        # ligand_receptor_energy = 0.0
        for ligand in self.ligands:
            ligands_intra_energy += self.calculator.intra_energy_of(ligand)
            # ligand_receptor_energy += self.calculator.inter_energy_between(ligand, self.receptor)
        # ligand - flex and ligand_i - ligand_j
        ligand_flex_energy = self.calculator.pairwise_energy_of(self.pairs["inter"])
        intra_energy = flex_intra_energy + ligands_intra_energy + receptor_intra_energy
        inter_energy = ligand_receptor_energy + ligand_flex_energy
        # FINAL 最终的
        if best_intra_score == 0.0:
            best_intra_score = intra_energy
        torsions = 0
        for ligand in self.ligands:
            torsions += ligand.torsion
        final_energy = self.calculator.g(intra_energy + inter_energy - best_intra_score, torsions)
        return (final_energy, intra_energy, inter_energy)


# 对接模型类
class DockingModelWithPoses(DockingModel):
    def __init__(self, weights: dict | None = None) -> None:
        super().__init__(weights)
        self.poses: dict[Atom, list[tuple[float, float, float]]] = {}
        self.pose_num = 0
        self.calculator_selector = 0
        self.calculator_controller = Calculator(weights)

    def set_poses(self, models: list[list[SimpleMolecule]]) -> None:
        self.validate()
        self.poses.clear()
        self.pose_num = len(models)
        for atom in self.flexible_atoms:
            self.poses[atom] = [atom.coordinate]
        for simple_molecules in models:
            for index, (simple_atoms, molecule_type) in enumerate(simple_molecules):
                for simple_atom in simple_atoms:
                    if molecule_type == "flex":
                        atom = self.flex[simple_atom.serial_num]
                    else:
                        atom = self.ligands[index][simple_atom.serial_num]
                    if not atom.immobile and atom.is_heavy_atom():
                        self.poses[atom].append(simple_atom.coordinate)

    # 多个pose的vina对接分数计算,以初始构象为标准
    def vina_with_poses(self, end_idx: int = 9999, print_mode=False, ml_mode=False) -> list[tuple[float, float, float]]:
        if print_mode:
            print("Vina对接分数计算开始.")
            print(f"当前计算模式: {'常规' if self.calculator_selector % 2 == 0 else '预先'}.")
        results: list[tuple[float, float, float]] = []
        best_intra_score = 0.0
        for i in range(self.pose_num+1):
            if i >= end_idx:
                break
            for atom, coordinate in self.poses.items():
                atom.coordinate = coordinate[i]
            energies = self.vina(best_intra_score, ml_mode)
            results.append(energies)
            if i == 0:
                best_intra_score = energies[1]
            if print_mode:
                if i == 0:
                    print("初始构象: ", end="")
                else:
                    print(f"第{i:02}个优化构象: ", end="")
                print(f"最终结合能:{energies[0]:7.3f}, 分子内能量:{energies[1]:6.3f}, 分子间能量:{energies[2]:7.3f}.")
        for atom, coordinate in self.poses.items():
            atom.coordinate = coordinate[0]
        return results
