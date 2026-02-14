"""
simulation.py文件归纳了仿真模型:
1.原子层面: 原子, 原子距离工具, 原子对, 键, 原子聚类
2.分子层面: 刚性分子, 分子
3.对接层面: 受体, 柔性侧链, 配体
4.基本信息传递：简单原子, 简单分子

Last update: 2026-02-14 by Junlin_hpc
version: 1.0.1 (从pkg_resources迁移到importlib.resources)
"""


# 导入区
import math
from collections import deque
from enum import IntEnum
from itertools import combinations
from typing import NamedTuple
from repro_vina.utility import load_configuration, SymmetricMatrix


# 常量区
# 原子记录样本(所有的record均为pdbqt格式)
ATOM_RECORD_TEMPLATE: str = "ATOM      0  ??? ??? ?   1       0.000   0.000   0.000  0.00  0.00     0.000 ? "
# 半径常量
ATOMIC_PARAMETE = load_configuration("atomic_parameter")
VDW_RADII: dict[str, float] = ATOMIC_PARAMETE["vdw radius"]
COVALENT_RADII: dict[str, float] = ATOMIC_PARAMETE["covalent radius"]
MAX_COVALENT_RADIUS = max(COVALENT_RADII.values())
EXTRA_METAL_COVALENT_RADIUS = 1.75
BEAD_RADIUS = 15.0
BOND_LENGTH_TOLERANCE = 1.1
# 类型常量
TYPE_GROUPS = load_configuration("type_groups")
METAL_TYPES = set(TYPE_GROUPS["metals"])
EXTRA_METAL_TYPES = set(TYPE_GROUPS["extra_metals"])
GLUEY_TYPES = set(TYPE_GROUPS["gluey"])
HYDROPHOBIC_TYPES = set(TYPE_GROUPS["hydrophobic"])
DONOR_TYPES = set(TYPE_GROUPS["donor"])
ACCEPTOR_TYPES = set(TYPE_GROUPS["acceptor"])
NON_HETEROATOMS = {"C", "H"}
GLUEY_PAIRS = {frozenset([f"G{i}", f"CG{i}"]) for i in range(4)}
# 几何约束类型
class GeometricConstraint(IntEnum):
    FIXED = 0
    ROTATABLE = 1
    VARIABLE = 6


# 模型区
# 1.原子类
class Atom:
    def __init__(self, atom_record: str = ATOM_RECORD_TEMPLATE) -> None:
        self.serial_num = int(atom_record[6:11])
        self.name = atom_record[12:16].strip()
        self.coordinate = (float(atom_record[30:38]), float(atom_record[38:46]), float(atom_record[46:54]))
        self.partial_charge = float(atom_record[68:76])
        self.types = {"autodock4": atom_record[77:79].strip(), "element": "?", "xscore": "?"}
        self.bonds: dict[Atom, Bond] = {}
        self.molecule: RigidMolecule | None = None
        self.immobile = False
        self.assign_element_type()

    def __str__(self) -> str:
        return f"Atom {self.name}({self.serial_num})"

    def assign_element_type(self) -> None:
        if self.is_extra_metal():
            self.types["element"] = self.types["autodock4"]
        else:
            self.types["element"] = TYPE_GROUPS["type_from_ad4_to_el"].get(self.types["autodock4"], "?")

    def bond_to(self, adjacent_atom: "Atom", bond: "Bond") -> None:
        self.bonds[adjacent_atom] = bond

    def belong_to(self, molecule: "RigidMolecule") -> None:
        self.molecule = molecule

    # 原子距离(强调源原子)
    def distance_to(self, target_atom: "Atom") -> float:
        return math.sqrt(sum(map(lambda i, j: math.pow(i-j, 2), self.coordinate, target_atom.coordinate)))

    # 共价半径(使用原子类型element)
    @property
    def covalent_radius(self) -> float:
        if self.types["element"] in COVALENT_RADII:
            return COVALENT_RADII[self.types["element"]]
        if self.is_extra_metal():
            return EXTRA_METAL_COVALENT_RADIUS
        # print(f"无法获取ad4类型[{self.types['autodock4']}]的共价半径,默认返回最大共价半径!")
        return MAX_COVALENT_RADIUS

    # 范德华半径(使用原子类型xscore)
    @property
    def vdw_radius(self) -> float:
        if self.types["xscore"] in VDW_RADII:
            return VDW_RADII[self.types["xscore"]]
        # print(f"无法获取xs类型[{self.types["xscore"]}]的范德华半径,默认返回0!")
        return 0.0

    # 重(非氢)原子判断(使用原子类型element)
    def is_heavy_atom(self) -> bool:
        return self.types["element"] != "H"

    # 杂(非碳氢)原子判断(使用原子类型element)
    def is_heteroatom(self) -> bool:
        return self.types["element"] not in NON_HETEROATOMS

    # 金属类型判断(使用原子类型element)
    def is_metal(self) -> bool:
        return self.types["element"] in METAL_TYPES

    # 额外金属类型判断(使用原子类型autodock4)
    def is_extra_metal(self) -> bool:
        return self.types["autodock4"] in EXTRA_METAL_TYPES

    # 粘连类型判断(使用原子类型autodock4)
    def is_gluey(self) -> bool:
        return self.types["autodock4"] in GLUEY_TYPES

    # 疏水性判断(使用原子类型xscore)
    def is_hydrophobic(self) -> bool:
        return self.types["xscore"] in HYDROPHOBIC_TYPES

    # 供体类型判断(使用原子类型xscore)
    def is_donor(self) -> bool:
        return self.types["xscore"] in DONOR_TYPES

    # 受体类型判断(使用原子类型xscore)
    def is_acceptor(self) -> bool:
        return self.types["xscore"] in ACCEPTOR_TYPES

    # 与供氢键的氢原子成键判断(使用原子类型autodock4)
    def is_bonded_to_HD(self) -> bool:  # pylint: disable=C0103
        for bonded_atom in self.bonds:
            if bonded_atom.types["autodock4"] == "HD":
                return True
        return False

    # 与杂原子成键判断
    def is_bonded_to_heteroatom(self) -> bool:
        for bonded_atom in self.bonds:
            if bonded_atom.is_heteroatom():
                return True
        return False

    # 获取指定跳数内的邻居
    def neighbors_within(self, bond_hop_limit: int) -> set["Atom"]:
        neighbor_atoms: set[Atom] = set()
        queue: deque[tuple[Atom, int]] = deque()
        queue.append((self, bond_hop_limit))
        while queue:
            atom, hop_limit = queue.popleft()
            neighbor_atoms.add(atom)
            if hop_limit <= 0:
                continue
            for bonded_atom in atom.bonds:
                if bonded_atom in neighbor_atoms or isinstance(bonded_atom.molecule, Receptor):
                    continue
                queue.append((bonded_atom, hop_limit-1))
        return neighbor_atoms

    # 校验
    @classmethod
    def simple_validata(cls, atom_record: str) -> bool:
        ART: str = "ATOM      0  ??? ??? ?   1       0.000   0.000   0.000  0.00  0.00     0.000 ? "
        try:
            if not atom_record.startswith("ATOM"):
                raise ValueError("The record is not atom's record.")
            serial_num = int(atom_record[6:11])
            name = atom_record[12:16].strip()
            coordinate = (float(atom_record[30:38]), float(atom_record[38:46]), float(atom_record[46:54]))
            partial_charge = float(atom_record[68:76])
            types = {"autodock4": atom_record[77:79].strip(), "element": "?", "xscore": "?"}
        except Exception as e:
            print(e)
            return False
        return True


# 2.原子距离工具
class AtomPositioner(SymmetricMatrix[GeometricConstraint]):
    def __init__(self, molecule = None, default_value = GeometricConstraint.VARIABLE) -> None:
        if isinstance(molecule, Molecule):
            super().__init__(molecule.size, default_value)
            self.analyse_position_type(molecule)

    def __getitem__(self, key: tuple) -> GeometricConstraint:
        key = self.ensure_key(key)
        value = self.data[key[0]][key[1]]
        assert value is not None
        return value

    def ensure_key(self, key: tuple[Atom, Atom]) -> tuple[int, int]:
        if all(isinstance(k, Atom) for k in key):
            new_key = (key[0].serial_num-1, key[1].serial_num-1)
            self.validate_key(new_key)
            return new_key
        raise TypeError("The key should follow the following format: (i: Atom, j: Atom).")

    # 分析距离类型(默认i<j)
    # 对于一般原子对(i, j),距离类型为固定的情况有三种:
    # (1)i和j共锚点; (2)i是j的锚点或者超锚点; (3)i的超锚点为0, 且为j的锚点或者超锚点.
    # 注: 对于键(a,b)且a<b, b的锚点设置为a, 而非a原先的锚点.
    def analyse_position_type(self, molecule: "Molecule") -> None:
        for atom_i, atom_j in combinations(molecule.atoms, 2):
            i, j = atom_i.serial_num, atom_j.serial_num
            if (molecule.anchor_of(i) == molecule.anchor_of(j)
                or i in {molecule.anchor_of(j), molecule.super_anchor_of(j)}
                or molecule.super_anchor_of(i) in ({0} & {molecule.anchor_of(j), molecule.super_anchor_of(j)})):
                self[i-1, j-1] = GeometricConstraint.FIXED
        for branch in molecule.branches:
            self[branch[0]-1, branch[1]-1] = GeometricConstraint.ROTATABLE


# 3.原子对
class AtomPair:
    def __init__(self, atom_i: Atom, atom_j: Atom) -> None:
        self.atoms = (atom_i, atom_j)
        self.ad4_types ={atom_i.types["autodock4"], atom_j.types["autodock4"]}

    def bonding(self, length: float, rotatable: bool) -> "Bond":
        bond = Bond(self, length, rotatable)
        self.atoms[0].bond_to(self.atoms[1], bond)
        self.atoms[1].bond_to(self.atoms[0], bond)
        return bond

    def distance(self) -> float:
        return self.atoms[0].distance_to(self.atoms[1])

    def geometric_relation(self, atom_positioner: AtomPositioner) -> GeometricConstraint:
        if self.atoms[0] is self.atoms[1]:
            return GeometricConstraint.FIXED
        if self.atoms[0].immobile and self.atoms[1].immobile:
            return GeometricConstraint.FIXED
        if any(isinstance(self.atoms[i].molecule, Receptor) and not self.atoms[1-i].immobile for i in range(2)):
            return GeometricConstraint.VARIABLE
        return atom_positioner[self.atoms]

    # 最佳距离(范德华半径决定)
    def optimal_distance(self) -> float:
        if self.atoms[0].is_gluey() or self.atoms[1].is_gluey():
            return 0.0
        return self.atoms[0].vdw_radius + self.atoms[1].vdw_radius

    # 最佳共价键长
    def optimal_covalent_bond_length(self) -> float:
        return self.atoms[0].covalent_radius + self.atoms[1].covalent_radius

    # 判断原子对是否在同一分子内
    def is_intramolecular(self) -> bool:
        return self.atoms[0].molecule is self.atoms[1].molecule

    # 判断原子对是否粘连
    def is_gluey(self) -> bool:
        return self.ad4_types in GLUEY_PAIRS

    # 判断原子对是否为供受对
    def is_donor_acceptor(self) -> bool:
        return any(self.atoms[i].is_donor() and self.atoms[1-i].is_acceptor() for i in range(2))

    # 判断原子对是否CG冲突
    def is_closure_clash(self) -> bool:
        if self.is_gluey():
            return False
        CG_existence = {"CG0": False, "CG1": False, "CG2": False, "CG3": False} # pylint: disable=C0103
        for atom in self.atoms:
            for neighbor in atom.neighbors_within(1):
                neighbor_ad4_type = neighbor.types["autodock4"]
                if neighbor_ad4_type in CG_existence:
                    if CG_existence[neighbor_ad4_type]:
                        return True
                    CG_existence[neighbor_ad4_type] = True
        return False

    #判断原子对是否存在G而不存在对应的CG
    def is_unmatched_closure_dummy(self) -> bool:
        return any(f"G{i}" in self.ad4_types and f"CG{i}" not in self.ad4_types for i in range(4))

    # 检验共价键——若原子对间存在其他原子,则共价键无效(False)
    def validate_covalent_bond(self, neighbor_atoms: list[Atom], atom_positioner: AtomPositioner) -> bool:
        distance_threshold = self.distance()
        for neighbor_atom in neighbor_atoms:
            if neighbor_atom in self.atoms:
                continue
            if all((pair := AtomPair(atom, neighbor_atom)).distance() < distance_threshold
                   and pair.geometric_relation(atom_positioner) != GeometricConstraint.VARIABLE
                   for atom in self.atoms):
                return False
        return True


# 4.键类
class Bond:
    def __init__(self, pair: AtomPair, length: float, rotatable: bool) -> None:
        self.atoms = pair.atoms
        self.length = length
        self.rotatable = rotatable

    def __str__(self) -> str:
        return (f"Bond{'(rotatable)' if self.rotatable else ''} "
                + f"with length {self.length} between: "
                + ", ".join(f"{atom}" for atom in self.atoms))


# 5.原子聚类类(为分配键准备)
class Bead:
    def __init__(self, central_atom: Atom, radius: float = BEAD_RADIUS) -> None:
        self.atoms = [central_atom]
        self.central_atom = central_atom
        self.radius = radius

    def cluster(self, target_atom: Atom) -> bool:
        if self.central_atom.distance_to(target_atom) <= self.radius:
            self.atoms.append(target_atom)
            return True
        return False


# 6.刚性分子类
class RigidMolecule:
    def __init__(self) -> None:
        self.atoms: list[Atom] = []
        self.__valid_atoms: list[Atom] = []
        self.atom_positioner = AtomPositioner()
        self.bonds: list[Bond] = []

    def __getitem__(self, key: int) -> Atom:
        if isinstance(key, int):
            return self.atoms[key-1]
        raise TypeError("Please use a single index to get atom.")

    @property
    def size(self) -> int:
        return len(self.atoms)

    @property
    def valid_atoms(self) -> list[Atom]:
        if len(self.__valid_atoms) == 0:
            self.__valid_atoms = [atom for atom in self.atoms if atom.is_heavy_atom()]
        return self.__valid_atoms

    def add_atom(self, atom: Atom) -> None:
        atom.belong_to(self)
        self.atoms.append(atom)

    # 根据原子聚类和定位分配键
    def assign_bonds(self, beads: list[Bead], atom_positioner: AtomPositioner) -> None:
        for atom in self.atoms:
            neighbor_atoms: list[Atom] = []
            bond_cutoff = BOND_LENGTH_TOLERANCE * (atom.covalent_radius+MAX_COVALENT_RADIUS)
            for bead in beads:
                if atom.distance_to(bead.central_atom) > bond_cutoff + bead.radius:
                    continue
                for bead_atom in bead.atoms:
                    pair = AtomPair(atom, bead_atom)
                    if pair.distance() <= bond_cutoff:
                        if pair.geometric_relation(atom_positioner) != GeometricConstraint.VARIABLE:
                            neighbor_atoms.append(bead_atom)
            for neighbor_atom in neighbor_atoms:
                if neighbor_atom in atom.bonds:
                    continue
                pair = AtomPair(atom, neighbor_atom)
                length = pair.optimal_covalent_bond_length()
                if (pair.distance() < BOND_LENGTH_TOLERANCE * length
                    and pair.validate_covalent_bond(neighbor_atoms, atom_positioner)):
                    rotatable = pair.geometric_relation(atom_positioner) == GeometricConstraint.ROTATABLE
                    self.bonds.append(pair.bonding(length, rotatable))

    # 分配原子类型xscore
    def assign_xscore_type(self) -> None:
        for atom in self.atoms:
            ad4_type = atom.types["autodock4"]
            el_type = atom.types["element"]
            is_donor = atom.is_bonded_to_HD() # 原子为金属类型时也为供体, 但此时将固定为Met_D类型, 故去除金属判断
            is_acceptor = ad4_type in {"OA", "NA"}
            if atom.is_metal() or atom.is_extra_metal():
                atom.types["xscore"] = "Met_D"
                continue
            match el_type:
                case "C":
                    atom.types["xscore"] = "C_P" if atom.is_bonded_to_heteroatom() else "C_H"
                    if ad4_type.startswith("CG"):
                        atom.types["xscore"] = f"{atom.types["xscore"]}_{ad4_type}"
                case "N" | "O":
                    atom.types["xscore"] = f"{el_type}_{'DA' if is_donor and is_acceptor else
                                                             'D' if is_donor else
                                                             'A' if is_acceptor else 'P'}"
                case "S" | "P":
                    atom.types["xscore"] = f"{el_type}_P"
                case "F" | "Cl" | "Br" | "I":
                    atom.types["xscore"] = f"{el_type}_H"
                case "Si" | "At":
                    atom.types["xscore"] = el_type
                case "Dummy":
                    atom.types["xscore"] = ad4_type
                case "?":
                    raise TypeError("Unidentified atom's element type.")

    # 活化
    def activate(self) -> None:
        beads: list[Bead] = []
        for atom in self.atoms:
            if not any(bead.cluster(atom) for bead in beads):
                beads.append(Bead(atom))
        self.assign_bonds(beads, self.atom_positioner)
        self.assign_xscore_type()


# 7.分子类
class Molecule(RigidMolecule):
    def __init__(self) -> None:
        super().__init__()
        self.branches: list[tuple[int, int]] = []
        self.anchors: list[int] = [0]
        self.anchor_pair: tuple[int, int] = (0, 0)
        self.pairs: list[AtomPair] = []

    @property
    def torsion(self) -> int:
        return len(self.branches)

    def anchor_of(self, atom_serial_num: int) -> int:
        return self.anchors[atom_serial_num]

    def super_anchor_of(self, atom_serial_num: int) -> int:
        return self.anchors[self.anchors[atom_serial_num]]

    def add_atom(self, atom: Atom) -> None:
        super().add_atom(atom)
        if atom.serial_num == self.anchor_pair[1]:
            self.anchors.append(self.anchor_pair[0])
        else:
            self.anchors.append(self.anchor_pair[1])

    def put_branch(self, parent_child: tuple[int, int]) -> None:
        self.branches.append(parent_child)
        self.anchor_pair = parent_child

    def validate_branch(self, parent_child: tuple[int, int]) -> None:
        for branch in self.branches:
            if parent_child == branch:
                self.anchor_pair = (self.anchors[parent_child[0]], parent_child[0])
                return
        raise ValueError("Unknown branch existes.")

    def validate_torsion(self, torsion: int) -> None:
        if self.torsion != torsion:
            raise ValueError("The number of rotatable bonds don't match the number of branches.")

    # 活化
    def activate(self) -> None:
        self.atom_positioner = AtomPositioner(self)
        super().activate()


# 8.受体类
class Receptor(RigidMolecule):
    @classmethod
    def from_records(cls, records: list[str]) -> "Receptor":
        receptor = cls()
        for record in records:
            if record.startswith("ATOM"):
                receptor.add_atom(Atom(record))
        return receptor

    def add_atom(self, atom: Atom) -> None:
        atom.immobile = True
        super().add_atom(atom)

    # 活化
    def activate(self, complement: "Flex") -> None: # type: ignore  # pylint: disable=W0221
        beads: list[Bead] = []
        for atom in (self.atoms + complement.atoms):
            if not any(bead.cluster(atom) for bead in beads):
                beads.append(Bead(atom))
        self.assign_bonds(beads, AtomPositioner(complement))
        self.assign_xscore_type()


# 9.配体类
class Ligand(Molecule):
    @classmethod
    def from_records(cls, records: list[str]) -> "Ligand":
        ligand = cls()
        for record in records:
            if record.startswith("ATOM"):
                ligand.add_atom(Atom(record))
            elif record.startswith("BRANCH"):
                ligand.put_branch((int(record[6:10]), int(record[10:14])))
            elif record.startswith("ENDBRANCH"):
                ligand.validate_branch((int(record[9:13]), int(record[13:17])))
            elif record.startswith("TORSDOF"):
                ligand.validate_torsion(int(record[7:10]))
        return ligand


# 10.柔性侧链类
class Flex(Molecule):
    @classmethod
    def from_records(cls, records: list[str]) -> "Flex":
        flex = cls()
        for record in records:
            if record.startswith("ATOM"):
                flex.add_atom(Atom(record))
            elif record.startswith("BRANCH"):
                flex.put_branch((int(record[6:10]), int(record[10:14])))
            elif record.startswith("ENDBRANCH"):
                flex.validate_branch((int(record[9:13]), int(record[13:17])))
            elif record.startswith("BEGIN_RES"):
                flex.anchor_pair = (0, 0)
        return flex

    def add_atom(self, atom: Atom) -> None:
        super().add_atom(atom)
        if 0 in {self.anchor_of(atom.serial_num), self.super_anchor_of(atom.serial_num)}:
            atom.immobile = True


# 11.简单原子类(只包含原子序号,坐标和原子类型)
class SimpleAtom(NamedTuple):
    serial_num: int
    coordinate: tuple[float, float, float]
    types: str

    @classmethod
    def from_record(cls, atom_record: str = ATOM_RECORD_TEMPLATE) -> "SimpleAtom":
        return cls(
            serial_num = int(atom_record[6:11]),
            coordinate = (float(atom_record[30:38]), float(atom_record[38:46]), float(atom_record[46:54])),
            types = atom_record[77:79].strip()
        )

    @classmethod
    def simplify(cls, atom: Atom) -> "SimpleAtom":
        return cls(
            serial_num = atom.serial_num,
            coordinate = atom.coordinate,
            types = atom.types["autodock4"]
        )


# 12.简单分子类(泛指配体和柔性侧链,只包含简单原子和分子类型)
class SimpleMolecule(NamedTuple):
    simple_atoms: list[SimpleAtom]
    molecule_type: str # ligand or flex

    @classmethod
    def simplify(cls, molecule: Molecule) -> "SimpleMolecule":
        return cls(
            simple_atoms = [SimpleAtom.simplify(atom) for atom in molecule.atoms],
            molecule_type = "ligand" if isinstance(molecule, Ligand) else "flex"
        )
