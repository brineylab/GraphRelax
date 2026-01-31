"""Binding energy (ddG) calculation for protein-protein interfaces."""

import io
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio.PDB import PDBIO, PDBParser, Select

from graphrelax.idealize import extract_ligands, restore_ligands
from graphrelax.interface import InterfaceResidue, identify_interface_residues

logger = logging.getLogger(__name__)


@dataclass
class BindingEnergyResult:
    """Results of binding energy calculation."""

    complex_energy: float = 0.0
    separated_energies: Dict[str, float] = field(default_factory=dict)
    binding_energy: float = 0.0
    energy_breakdown: Dict[str, float] = field(default_factory=dict)
    interface_residues: List[InterfaceResidue] = field(default_factory=list)
    interface_energy: Optional[float] = None


class _ChainSelector(Select):
    """BioPython Select subclass to filter to specific chains."""

    def __init__(self, chain_ids: list):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.id in self.chain_ids


def extract_chain(pdb_string: str, chain_ids: list) -> str:
    """
    Extract specific chains from a multi-chain PDB.

    Args:
        pdb_string: PDB file contents
        chain_ids: List of chain IDs to extract

    Returns:
        PDB string containing only the specified chains
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", io.StringIO(pdb_string))

    pdb_io = PDBIO()
    pdb_io.set_structure(structure)

    output = io.StringIO()
    pdb_io.save(output, _ChainSelector(chain_ids))
    return output.getvalue()


def _get_interface_chain_groups(
    chain_pairs: List[Tuple[str, str]],
) -> List[List[str]]:
    """
    Group chains into sides of the interface.

    For pairs like [(H, A), (L, A)], returns [[H, L], [A]].
    Uses a union-find approach to group chains that appear on the same
    side of any interface pair.

    Args:
        chain_pairs: List of (chain_a, chain_b) tuples

    Returns:
        List of chain groups (each group is a list of chain IDs)
    """
    # Collect all unique chains
    all_chains = set()
    for a, b in chain_pairs:
        all_chains.add(a)
        all_chains.add(b)

    # For simple cases (single pair), just return both sides
    if len(chain_pairs) == 1:
        a, b = chain_pairs[0]
        return [[a], [b]]

    # For multiple pairs, group chains that always appear on the same side.
    # Chains on side 1 (first element of pairs) vs side 2 (second element).
    side1 = set()
    side2 = set()
    for a, b in chain_pairs:
        side1.add(a)
        side2.add(b)

    # If sides overlap, fall back to individual chains
    if side1 & side2:
        return [[c] for c in sorted(all_chains)]

    return [sorted(side1), sorted(side2)]


def calculate_binding_energy(
    pdb_string: str,
    relaxer: "Relaxer",  # noqa: F821
    chain_pairs: Optional[List[Tuple[str, str]]] = None,
    distance_cutoff: float = 8.0,
    relax_separated: bool = True,
) -> BindingEnergyResult:
    """
    Calculate binding energy by comparing complex and separated chain energies.

    Workflow:
    1. Get energy of the complex (already relaxed)
    2. Identify interface residues
    3. Extract each side of the interface to separate PDBs
    4. Optionally relax separated chains
    5. Calculate ddG = E_complex - sum(E_separated)

    Args:
        pdb_string: Complex PDB structure (should already be relaxed)
        relaxer: Configured Relaxer instance
        chain_pairs: Chain pairs to analyze (auto-detect if None)
        distance_cutoff: Interface distance cutoff (angstroms)
        relax_separated: Whether to relax separated chains

    Returns:
        BindingEnergyResult with energies and interface info
    """
    # Step 1: Get complex energy
    logger.info("  Computing complex energy...")
    complex_breakdown = relaxer.get_energy_breakdown(pdb_string)
    complex_energy = complex_breakdown.get("total_energy", 0.0)

    # Step 2: Identify interface residues
    interface_info = identify_interface_residues(
        pdb_string,
        distance_cutoff=distance_cutoff,
        chain_pairs=chain_pairs,
    )

    if not interface_info.interface_residues:
        logger.warning("  No interface residues found - cannot compute ddG")
        return BindingEnergyResult(
            complex_energy=complex_energy,
            energy_breakdown=complex_breakdown,
        )

    # Step 3: Group chains into sides and extract
    chain_groups = _get_interface_chain_groups(interface_info.chain_pairs)
    logger.info(
        f"  Separating chains into {len(chain_groups)} groups: "
        + ", ".join(f"[{'+'.join(g)}]" for g in chain_groups)
    )

    # Extract ligands first - they should not be included in separated chains
    protein_pdb, ligand_lines = extract_ligands(pdb_string)

    # Step 4: Compute separated chain energies
    separated_energies = {}
    for group in chain_groups:
        group_label = "+".join(group)
        logger.info(f"  Computing energy for chain(s) {group_label}...")

        chain_pdb = extract_chain(protein_pdb, group)

        if relax_separated:
            # Relax the separated chain(s)
            try:
                relaxed_pdb, relax_info, _ = relaxer.relax(chain_pdb)
                chain_breakdown = relaxer.get_energy_breakdown(relaxed_pdb)
                chain_energy = chain_breakdown.get("total_energy", 0.0)
                logger.info(
                    f"    Chain(s) {group_label}: "
                    f"E_init={relax_info['initial_energy']:.2f}, "
                    f"E_final={chain_energy:.2f}"
                )
            except Exception as e:
                logger.warning(
                    f"    Failed to relax chain(s) {group_label}: {e}"
                )
                # Fall back to unrelaxed energy
                chain_breakdown = relaxer.get_energy_breakdown(chain_pdb)
                chain_energy = chain_breakdown.get("total_energy", 0.0)
        else:
            chain_breakdown = relaxer.get_energy_breakdown(chain_pdb)
            chain_energy = chain_breakdown.get("total_energy", 0.0)

        separated_energies[group_label] = chain_energy

    # Step 5: Calculate ddG
    total_separated = sum(separated_energies.values())
    binding_energy = complex_energy - total_separated

    logger.info(
        f"  ddG = {binding_energy:.2f} kcal/mol "
        f"(complex: {complex_energy:.2f}, "
        f"separated: {total_separated:.2f})"
    )

    return BindingEnergyResult(
        complex_energy=complex_energy,
        separated_energies=separated_energies,
        binding_energy=binding_energy,
        energy_breakdown=complex_breakdown,
        interface_residues=interface_info.interface_residues,
    )
