from __future__ import annotations

import os
import warnings
import argparse
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem
from openff.toolkit import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange
from tqdm import tqdm

# ENV
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# LOGGING
def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

# Types
MolGroupKey = Tuple[str, str]
MolGroup = Dict[MolGroupKey, List[Chem.Mol]]


# Grouping
def group_molecules(sdf_path: str) -> MolGroup:
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    groups: MolGroup = defaultdict(list)

    for idx, mol in enumerate(supplier):
        if mol is None:
            logger.warning(f"Skipping invalid molecule at index {idx}")
            continue

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.warning(f"Sanitize failed at {idx}: {e}")
            continue

        smiles = Chem.MolToSmiles(mol)
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"

        groups[(name, smiles)].append(mol)

    logger.info(f"Grouped into {len(groups)} unique molecules")
    return groups


# CORE
def build_interchange(mol: Chem.Mol, mol_name: str) -> tuple[Molecule, Interchange]:
    force_field = ForceField("openff-2.1.0.offxml")

    off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    off_mol.name = mol_name

    topology = off_mol.to_topology()

    interchange = Interchange.from_smirnoff(
        force_field=force_field,
        topology=topology,
    )

    return off_mol, interchange


def compute_box(coords_nm: np.ndarray) -> np.ndarray:
    min_c = coords_nm.min(axis=0)
    max_c = coords_nm.max(axis=0)

    mol_size = np.max(max_c - min_c) + 2.0
    mol_size = max(mol_size, 2.0)

    return np.eye(3) * mol_size


def extract_coords_nm(rdmol: Chem.Mol) -> np.ndarray:
    conf = rdmol.GetConformer()

    coords = np.array([
        conf.GetAtomPosition(i) for i in range(rdmol.GetNumAtoms())
    ])

    return coords / 10.0


# WORKER
def process_mol_group(args: Tuple[MolGroupKey, List[Chem.Mol], str]) -> None:
    (mol_name, smiles), mols, output_dir = args

    logger.info(f"Processing: {mol_name} ({len(mols)} conformers)")

    base_mol = mols[0]
    Chem.SanitizeMol(base_mol)

    off_mol, interchange = build_interchange(base_mol, mol_name)

    mol_dir = os.path.join(output_dir, mol_name)
    os.makedirs(mol_dir, exist_ok=True)

    # TOPOLOGY
    top_path = os.path.join(mol_dir, "topol.top")
    interchange.to_top(top_path)

    # CONFORMERS
    for i, rdmol in enumerate(mols):
        try:
            Chem.SanitizeMol(rdmol)
            coords_nm = extract_coords_nm(rdmol)

            interchange.positions = coords_nm
            interchange.box = compute_box(coords_nm)

            gro_path = os.path.join(mol_dir, f"conf_{i}.gro")
            interchange.to_gro(gro_path)

        except Exception as e:
            logger.error(f"Failed conformer {i} in {mol_name}: {e}")

    logger.info(f"Done: {mol_name}")


# PARALLEL
def process_sdf_parallel(
    sdf_path: str,
    output_dir: str,
    n_workers: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    groups = group_molecules(sdf_path)

    tasks = [
        ((name, smiles), mols, output_dir)
        for (name, smiles), mols in groups.items()
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(
            executor.map(process_mol_group, tasks),
            total=len(tasks),
            desc="Processing molecules"
        ))


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SDF to GROMACS"
    )

    parser.add_argument("--sdf", required=True, help="Input SDF file")
    parser.add_argument("--out", default="ligands", help="Output dir")
    parser.add_argument("--nproc", type=int, default=1, help="Processes")
    parser.add_argument("--log", default="run.log", help="Log file")

    return parser.parse_args()


# MAIN
def main() -> None:
    args = parse_args()

    setup_logging(args.log)

    logger.info(f"Starting processing: {args.sdf}")

    process_sdf_parallel(
        sdf_path=args.sdf,
        output_dir=args.out,
        n_workers=args.nproc,
    )

    logger.info("All done")


if __name__ == "__main__":
    main()
