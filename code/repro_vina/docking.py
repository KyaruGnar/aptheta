"""
docking.py文件归纳了对接模型:
1.能量计算模型: Vina各项能量计算项, Vina对接计算逻辑, 基于原子类型的预先计算
2.分子对接模型: 基本对接(标准输入)模型, 重对接(标准结构和对接结果作为输入)模型

Last update: 2026-02-14 by Junlin_hpc
version: 1.0.1 (从pkg_resources迁移到importlib.resources)
"""


# 导入区
import math
import sys
from importlib import resources
from itertools import combinations, combinations_with_replacement, product
from repro_vina.utility import load_configuration, SymmetricMatrix, HardSigmod
from repro_vina.parser import parse_receptor_pdbqts, parse_ligand_pdbqts, parse_output_pdbqt_simply
from repro_vina.simulation import (TYPE_GROUPS, GeometricConstraint,
                                   Atom, AtomPair, RigidMolecule, Molecule, Flex, Receptor, Ligand, SimpleMolecule)


# 常量区
CURL_THRESHOLD = 1000.0
FLOAT_EPSILON = sys.float_info.epsilon
FLOAT_MAXIMUM = sys.float_info.max
FLOAT_MINIMUM = sys.float_info.min
VINA_WEIGHTS = load_configuration("weights")["vina"]


# 一. 能量计算模型
# 能量独立计算逻辑
# 1.阈值限制方法: 当计算的能量大于0时，通过平滑变换将能量限制在阈值以内
def curl(enery: float, threshold: float = CURL_THRESHOLD) -> float:
    if enery > 0 and threshold < 0.1 * FLOAT_MAXIMUM:
        tmp = 0 if threshold < FLOAT_EPSILON else (threshold / (threshold+enery))
        enery *= tmp
    return enery


# 2.势能基类
class Potential:
    def calculate(self, pair: AtomPair, r: float) -> float: # pylint: disable=W0613
        return 0.0


# 3.高斯项计算类
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


# 4.斥力项计算类
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


# 5.疏水项计算类
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


# 6.氢键项计算类
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


# 7.线性吸引项计算类
class LinearAttraction(Potential):
    def __init__(self, cutoff: float) -> None:
        self.cutoff = cutoff

    def calculate(self, pair: AtomPair, r: float) -> float:
        if r >= self.cutoff:
            return 0.0
        if pair.is_gluey():
            return r
        return 0.0


# 能量组合计算逻辑
# 8.能量运算类
class Calculator:
    def __init__(self) -> None:
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
        self.max_cutoff = 20.0

    # 原子对能量计算
    def paired_energy_of(self, pair: AtomPair, r: float | None = None) -> float:
        energy = 0.0
        if r is None:
            r = pair.distance()
        for (term_name, term_class) in self.calculators.items():
            energy += VINA_WEIGHTS[term_name] * term_class.calculate(pair, r)
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
            raise TypeError("The molecule_i should be the instance of Molecule.")
        energy = 0.0
        for atom_i in molecule_i.valid_atoms:
            if atom_i.immobile:
                continue
            energy += curl(sum(self.paired_energy_of(AtomPair(atom_i, atom_j)) for atom_j in molecule_j.valid_atoms))
        return energy

    # 一个与旋转自由度有关的单调递增函数,使打分结果间的差异更明显
    def g(self, energy, torsion):
        return energy / (1 + VINA_WEIGHTS["Rot"]*torsion)


# 能量组合预计算逻辑
# 9.预计算单元类
class PrecalculateUnit:
    def __init__(self, factor: float, split_length: int) -> None:
        self.factor = factor
        self.length = split_length
        self.fast_results: list[float] = [0.0] * self.length

    def init_fast(self, results: list[float]) -> None:
        for i in range(self.length):
            successor = 0.0 if i + 1 >= self.length else results[i+1]
            self.fast_results[i] = (results[i]+successor) / 2

    def calculate_fast(self, r2: float) -> float:
        i = int(self.factor * r2)
        assert i < len(self.fast_results), f"索引{i}超出预设范围,请检查预计算类中的split_length参数[{self.length}]!"
        return self.fast_results[i]


# 10.预计算类
class Precalculator(Calculator):
    def __init__(self, threshold: float = FLOAT_MAXIMUM, factor: float = 32) -> None:
        super().__init__()
        self.threshold = threshold
        self.factor = factor
        self.split_length = int(factor * self.max_cutoff**2) + 3
        self.r_distribution = [math.pow(i/self.factor, 0.5) for i in range(self.split_length)]
        self.xscore_types: dict[str, int] = {xs: i for i, xs in enumerate(TYPE_GROUPS["xscore"])}
        self.results = SymmetricMatrix[PrecalculateUnit](len(TYPE_GROUPS["xscore"]))
        self.precalculated = False

    def precalculate(self) -> "Precalculator":
        if self.precalculated:
            return self
        pair = AtomPair(Atom(), Atom())
        for (type_i, i), (type_j, j) in combinations_with_replacement(self.xscore_types.items(), 2):
            unit = PrecalculateUnit(self.factor, self.split_length)
            for atom, xs in tuple(zip(pair.atoms, (type_i, type_j))):
                atom.types["xscore"] = xs
            results = [0.0] * self.split_length
            for k in range(self.split_length):
                results[k] = min(self.threshold, super().paired_energy_of(pair, self.r_distribution[k]))
            unit.init_fast(results)
            self.results[i, j] = unit
        self.precalculated = True
        return self

    def calculate(self, pair: AtomPair, r2: float) -> float:
        i, j = tuple(self.xscore_types[atom.types["xscore"]] for atom in pair.atoms)
        unit = self.results[i, j]
        if unit is None:
            raise AttributeError("The precalculate unit doesn't exist.")
        return unit.calculate_fast(r2)

    # 两原子能量计算
    def paired_energy_of(self, pair: AtomPair, r: float | None = None, r2: float | None = None) -> float:
        if r is None and r2 is None:
            r = pair.distance()
        if r:
            r2 = math.pow(r, 2)
        assert r2 is not None
        if r2 > self.max_cutoff**2:
            return 0.0
        return self.calculate(pair, r2)


# 二. 分子对接模型
# 1.基础对接模型
class DockingModel:
    def __init__(self) -> None:
        self.receptor = Receptor()
        self.flex = Flex()
        self.ligands: list[Ligand] = []
        self.flexible_atoms: list[Atom] = []
        self.pairs: dict[str, list[AtomPair]] = {"gluey": [], "inter": [], "other": []}
        self.calculator = Calculator()

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
            raise ValueError("The docking model's structure is incomplete.")

    # vina分数计算
    def vina(self, best_intra_score=0.0) -> tuple[float, float, float]:
        self.validate()
        # flex - receptor
        receptor_intra_energy = self.calculator.inter_energy_between(self.flex, self.receptor)
        # flex_i - flex_i and flex_i - flex_j
        flex_intra_energy = self.calculator.pairwise_energy_of(self.pairs["other"])
        # ligand_i - ligand_i / ligand - receptor
        ligands_intra_energy = 0.0
        ligand_receptor_energy = 0.0
        for ligand in self.ligands:
            ligands_intra_energy += self.calculator.intra_energy_of(ligand)
            ligand_receptor_energy += self.calculator.inter_energy_between(ligand, self.receptor)
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


# 2.重对接模型
class DockingModelWithPoses(DockingModel):
    def __init__(self) -> None:
        super().__init__()
        self.poses: dict[Atom, list[tuple[float, float, float]]] = {}
        self.pose_num = 0
        self.calculator_selector = 0
        self.calculator_controller = [Calculator(), Precalculator()]

    # 在常规计算模式和预先计算模式中切换,初始为计算模式
    def switch_calculator(self) -> None:
        self.calculator_selector += 1
        if self.calculator_selector % 2 == 0:
            self.calculator = self.calculator_controller[0]
            print("启用常规计算模式.")
        else:
            self.calculator = self.calculator_controller[1].precalculate() # type: ignore
            print("启用预先计算模式.")

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
    def vina_with_poses(self, end_idx: int = 9999, benchmark_idx: int = 0,
                        print_mode: bool = False) -> list[tuple[float, float, float]]:
        if print_mode:
            print("Vina对接分数计算开始.")
            print(f"当前能量基准位姿: {benchmark_idx}.")
            print(f"当前计算模式: {'常规' if self.calculator_selector % 2 == 0 else '预先'}.")
        results: list[tuple[float, float, float]] = []
        best_intra_score = 0.0
        for i in range(self.pose_num+1):
            if i >= end_idx:
                break
            for atom, coordinate in self.poses.items():
                atom.coordinate = coordinate[i]
            energies = self.vina(best_intra_score)
            results.append(energies)
            if i == benchmark_idx:
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


# 三.对接工作流
# 1.以DockingModel为模型的工作流
def workflow_with_DockingModel(receptor: str | None = None, ligands: list[str] | None = None, # pylint: disable=C0103
                               flex: str | None = None, example: int = -1):
    """
    examples:\n
    0: receptor: 2c0k的A链; ligands: 仅HEM; flex: 无.\n
    1: receptor: 2c0k的A链; ligands: HEM和OXY; flex: 无.\n
    2: receptor: 2c0k但不包含两个柔性侧链; ligands: HEM和OXY; flex: 两个柔性侧链.
    """
    # 参考设置
    examples = [
        ("2c0k_A.pdbqt", ["2c0k_C_HEM.pdbqt"], None),
        ("2c0k_A.pdbqt", ["2c0k_C_HEM.pdbqt", "2c0k_D_OXY.pdbqt"], None),
        ("2c0k_without_2f.pdbqt", ["2c0k_C_HEM.pdbqt", "2c0k_D_OXY.pdbqt"], "2f_in_2c0k.pdbqt"),
    ]
    if example > -1:
        file_location = resources.files(__package__).joinpath("example")
        receptor = f"{file_location}/{examples[example][0]}"
        ligands = []
        for ligand in examples[example][1]:
            ligands.append(f"{file_location}/{ligand}")
        if examples[example][2] is not None:
            flex = f"{file_location}/{examples[example][2]}"
    if receptor is None or ligands is None:
        raise ValueError("If no examples are needed, make sure the receptor and the ligands are not None.")
    # 模型构建
    model = DockingModel()
    model.set_receptor(*parse_receptor_pdbqts(receptor, flex))
    model.set_ligands(parse_ligand_pdbqts(ligands))
    # 设置运算类进行运算
    energies = model.vina()
    print(f"最终结合能:{energies[0]:7.3f}, 分子内能量:{energies[1]:6.3f}, 分子间能量:{energies[2]:7.3f}.")
    # 流程结束
    print("流程结束.")

# 2.以DockingModel为模型的工作流
def workflow_with_DockingModelWithPoses(receptor: str | None = None, ligands: list[str] | None = None, # pylint: disable=C0103+R0917
                                        flex: str | None = None, output: str | None = None,
                                        benchmark_idx: int = 0, precalculate: bool = False, example: int = -1):
    """
    examples:\n
    0: receptor: 2c0k的A链; ligands: 仅HEM; flex: 无; output: 对接结果; 能量基准位姿: 0; 启用预计算: 否.\n
    1: receptor: 2c0k的A链; ligands: 仅HEM; flex: 无; output: 对接结果; 能量基准位姿: 1; 启用预计算: 否.\n
    2: receptor: 2c0k的A链; ligands: 仅HEM; flex: 无; output: 对接结果; 能量基准位姿: 1; 启用预计算: 是.\n
    3: receptor: 2c0k但不包含两个柔性侧链; ligands: HEM和OXY; flex: 两个柔性侧链; output: 对接结果; 能量基准位姿: 1; 启用预计算: 是.
    """
    # 参考设置
    examples = [
        ("2c0k_A.pdbqt", ["2c0k_C_HEM.pdbqt"], None, "2c0k_C_HEM_out.pdbqt", 0, False),
        ("2c0k_A.pdbqt", ["2c0k_C_HEM.pdbqt"], None, "2c0k_C_HEM_out.pdbqt", 1, False),
        ("2c0k_A.pdbqt", ["2c0k_C_HEM.pdbqt"], None, "2c0k_C_HEM_out.pdbqt", 1, True),
        ("2c0k_without_2f.pdbqt", ["2c0k_D_OXY.pdbqt", "2c0k_C_HEM.pdbqt"], "2f_in_2c0k.pdbqt",
         "9m4w2l2f.pdbqt", 1, True),
    ]
    if example > -1:
        file_location = resources.files(__package__).joinpath("example")
        receptor = f"{file_location}/{examples[example][0]}"
        ligands = []
        for ligand in examples[example][1]:
            ligands.append(f"{file_location}/{ligand}")
        if examples[example][2] is not None:
            flex = f"{file_location}/{examples[example][2]}"
        output = f"{file_location}/{examples[example][3]}"
        benchmark_idx = examples[example][4]
        precalculate = examples[example][5]
    if receptor is None or ligands is None or output is None:
        raise ValueError("If no examples are needed, make sure the receptor, the ligands and the output are not None.")
    # 模型构建
    model = DockingModelWithPoses()
    model.set_receptor(*parse_receptor_pdbqts(receptor, flex))
    model.set_ligands(parse_ligand_pdbqts(ligands))
    if precalculate:
        model.switch_calculator()
    # 实际计算设置
    model.set_poses(parse_output_pdbqt_simply(output))
    # 设置运算类进行运算
    model.vina_with_poses(benchmark_idx=benchmark_idx, print_mode=True)
    # 流程结束
    print("流程结束.")
