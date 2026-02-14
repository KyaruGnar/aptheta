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