"""Post-minimization geometry validation for relaxed structures."""

import io
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

# Add vendored LigandMPNN to path for OpenFold imports
LIGANDMPNN_PATH = Path(__file__).parent / "LigandMPNN"
if str(LIGANDMPNN_PATH) not in sys.path:
    sys.path.insert(0, str(LIGANDMPNN_PATH))

from openfold.np import residue_constants as rc  # noqa: E402

logger = logging.getLogger(__name__)

# Standard backbone bond lengths (Engh & Huber reference geometry)
# These are intra-residue values not available in residue_constants
BACKBONE_BOND_LENGTHS = {
    ("N", "CA"): (1.458, 0.019),  # (ideal length, stddev) in Angstroms
    ("CA", "C"): (1.525, 0.021),
    ("C", "O"): (1.229, 0.019),
}

# Tolerance multiplier for backbone bonds (in Angstroms)
BOND_LENGTH_TOLERANCE = 0.02

# Tolerance factor for bond angles (matching OpenFold's violation_tolerance_factor)
ANGLE_TOLERANCE_FACTOR = 12.0

# Steric clash overlap tolerance in Angstroms (matching OpenFold)
CLASH_OVERLAP_TOLERANCE = 1.5

# Omega angle deviation tolerance from 180 degrees (for non-proline)
OMEGA_DEVIATION_TOLERANCE = 30.0

# Standard amino acid 3-letter codes
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}


@dataclass
class GeometryReport:
    """Report of geometry validation results."""

    n_residues: int = 0
    bond_length_violations: list = field(default_factory=list)
    bond_angle_violations: list = field(default_factory=list)
    steric_clashes: list = field(default_factory=list)
    omega_violations: list = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        return self.violation_count > 0

    @property
    def violation_count(self) -> int:
        return (
            len(self.bond_length_violations)
            + len(self.bond_angle_violations)
            + len(self.steric_clashes)
            + len(self.omega_violations)
        )

    @property
    def summary(self) -> str:
        parts = []
        if self.bond_length_violations:
            parts.append(
                f"{len(self.bond_length_violations)} bond length"
            )
        if self.bond_angle_violations:
            parts.append(
                f"{len(self.bond_angle_violations)} bond angle"
            )
        if self.steric_clashes:
            parts.append(f"{len(self.steric_clashes)} steric clash")
        if self.omega_violations:
            parts.append(f"{len(self.omega_violations)} omega angle")
        if not parts:
            return f"No violations ({self.n_residues} residues)"
        return (
            f"{self.violation_count} total violations "
            f"({', '.join(parts)}) in {self.n_residues} residues"
        )


def _get_atom_coord(residue, atom_name):
    """Get coordinates for a named atom in a residue, or None if missing."""
    if atom_name in residue:
        return residue[atom_name].get_vector().get_array()
    return None


def _atom_distance(coord1, coord2):
    """Euclidean distance between two coordinate arrays."""
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


def _dihedral_angle(p0, p1, p2, p3):
    """Compute dihedral angle in degrees from four points."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1
    b1 /= np.linalg.norm(b1)

    # Project b0 and b2 onto plane perpendicular to b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.degrees(np.arctan2(y, x))


def _angle_between(v1, v2, v3):
    """Compute angle in degrees at v2 between v1-v2-v3."""
    a = v1 - v2
    b = v3 - v2
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def check_bond_lengths(structure) -> list:
    """
    Check backbone bond lengths against ideal values.

    Checks intra-residue bonds (N-CA, CA-C, C=O) against Engh & Huber values
    and inter-residue peptide bonds (C-N) against OpenFold constants.

    Returns list of (chain_id, resnum, resname, atom1, atom2, observed, expected)
    tuples for violations.
    """
    violations = []

    for model in structure:
        for chain in model:
            residues = [
                r for r in chain.get_residues()
                if r.get_resname() in STANDARD_RESIDUES
            ]

            for i, residue in enumerate(residues):
                chain_id = chain.id
                resnum = residue.id[1]
                resname = residue.get_resname()

                # Check intra-residue backbone bonds
                for (a1, a2), (ideal, _stddev) in BACKBONE_BOND_LENGTHS.items():
                    c1 = _get_atom_coord(residue, a1)
                    c2 = _get_atom_coord(residue, a2)
                    if c1 is None or c2 is None:
                        continue
                    dist = _atom_distance(c1, c2)
                    if abs(dist - ideal) > BOND_LENGTH_TOLERANCE:
                        violations.append((
                            chain_id, resnum, resname,
                            a1, a2, float(dist), ideal,
                        ))

                # Check inter-residue peptide bond (C of this residue to N
                # of next)
                if i < len(residues) - 1:
                    next_res = residues[i + 1]
                    c_coord = _get_atom_coord(residue, "C")
                    n_coord = _get_atom_coord(next_res, "N")
                    if c_coord is None or n_coord is None:
                        continue
                    dist = _atom_distance(c_coord, n_coord)

                    # Use proline-specific values if next residue is PRO
                    is_pro = next_res.get_resname() == "PRO"
                    idx = 1 if is_pro else 0
                    ideal = rc.between_res_bond_length_c_n[idx]
                    if abs(dist - ideal) > BOND_LENGTH_TOLERANCE:
                        violations.append((
                            chain_id, resnum, resname,
                            "C", "N(+1)",
                            float(dist), ideal,
                        ))

    return violations


def check_bond_angles(structure) -> list:
    """
    Check backbone bond angles against ideal values.

    Uses OpenFold cosine angle constants for inter-residue angles
    (CA-C-N and C-N-CA). Tolerance is stddev * ANGLE_TOLERANCE_FACTOR.

    Returns list of (chain_id, resnum, resname, atoms, observed_deg,
    expected_deg) tuples for violations.
    """
    violations = []

    # Convert cosine angle constants to degrees
    # CA-C-N angle
    ca_c_n_cos_mean = rc.between_res_cos_angles_ca_c_n[0]
    ca_c_n_cos_std = rc.between_res_cos_angles_ca_c_n[1]
    ca_c_n_mean_deg = np.degrees(np.arccos(ca_c_n_cos_mean))

    # C-N-CA angle
    c_n_ca_cos_mean = rc.between_res_cos_angles_c_n_ca[0]
    c_n_ca_cos_std = rc.between_res_cos_angles_c_n_ca[1]
    c_n_ca_mean_deg = np.degrees(np.arccos(c_n_ca_cos_mean))

    # Compute tolerance in degrees from cosine stddev
    # Use derivative: d(angle)/d(cos) = -1/sin(angle)
    ca_c_n_tol_deg = abs(
        ca_c_n_cos_std * ANGLE_TOLERANCE_FACTOR
        / np.sin(np.arccos(ca_c_n_cos_mean))
    ) * (180.0 / np.pi)
    c_n_ca_tol_deg = abs(
        c_n_ca_cos_std * ANGLE_TOLERANCE_FACTOR
        / np.sin(np.arccos(c_n_ca_cos_mean))
    ) * (180.0 / np.pi)

    for model in structure:
        for chain in model:
            residues = [
                r for r in chain.get_residues()
                if r.get_resname() in STANDARD_RESIDUES
            ]

            for i in range(len(residues) - 1):
                res_i = residues[i]
                res_j = residues[i + 1]
                chain_id = chain.id
                resnum = res_i.id[1]
                resname = res_i.get_resname()

                ca_i = _get_atom_coord(res_i, "CA")
                c_i = _get_atom_coord(res_i, "C")
                n_j = _get_atom_coord(res_j, "N")
                ca_j = _get_atom_coord(res_j, "CA")

                # Check CA(i)-C(i)-N(i+1) angle
                if ca_i is not None and c_i is not None and n_j is not None:
                    angle = _angle_between(ca_i, c_i, n_j)
                    if abs(angle - ca_c_n_mean_deg) > ca_c_n_tol_deg:
                        violations.append((
                            chain_id, resnum, resname,
                            "CA-C-N(+1)",
                            float(angle), ca_c_n_mean_deg,
                        ))

                # Check C(i)-N(i+1)-CA(i+1) angle
                if c_i is not None and n_j is not None and ca_j is not None:
                    angle = _angle_between(c_i, n_j, ca_j)
                    next_resnum = res_j.id[1]
                    next_resname = res_j.get_resname()
                    if abs(angle - c_n_ca_mean_deg) > c_n_ca_tol_deg:
                        violations.append((
                            chain_id, next_resnum, next_resname,
                            "C(-1)-N-CA",
                            float(angle), c_n_ca_mean_deg,
                        ))

    return violations


def check_steric_clashes(structure) -> list:
    """
    Check inter-residue atom-atom distances for steric clashes.

    Uses van der Waals radii from OpenFold residue_constants. Overlap
    tolerance matches OpenFold's clash_overlap_tolerance (1.5 A).
    Skips bonded pairs (1-2 and 1-3 neighbors).

    Returns list of (res1_info, atom1, res2_info, atom2, distance,
    min_allowed) tuples for clashes.
    """
    clashes = []

    for model in structure:
        for chain in model:
            residues = [
                r for r in chain.get_residues()
                if r.get_resname() in STANDARD_RESIDUES
            ]

            for i in range(len(residues)):
                for j in range(i + 2, len(residues)):
                    res_i = residues[i]
                    res_j = residues[j]

                    # For adjacent residues (j == i+1 is skipped above),
                    # j == i+2 means we still need to check but skip
                    # 1-3 bonded pairs (C-N-CA across peptide bond)
                    is_near = (j == i + 2)

                    for atom_i in res_i.get_atoms():
                        elem_i = atom_i.element.strip()
                        if elem_i not in rc.van_der_waals_radius:
                            continue
                        if elem_i == "H":
                            continue

                        for atom_j in res_j.get_atoms():
                            elem_j = atom_j.element.strip()
                            if elem_j not in rc.van_der_waals_radius:
                                continue
                            if elem_j == "H":
                                continue

                            # Skip 1-3 bonded pair: C(i)-N(i+1)-CA(i+1)
                            # means C of res i and CA of res i+1 are 1-3
                            # bonded. For j == i+2, N of res j could be
                            # 1-3 bonded with C of res j-1.
                            if is_near:
                                # For residues separated by 1, the only
                                # cross-residue 1-3 pair is already excluded
                                # by skipping j == i+1. For j == i+2, no
                                # cross-residue 1-3 bonds exist.
                                pass

                            dist = atom_i - atom_j
                            vdw_sum = (
                                rc.van_der_waals_radius[elem_i]
                                + rc.van_der_waals_radius[elem_j]
                            )
                            min_allowed = vdw_sum - CLASH_OVERLAP_TOLERANCE
                            if dist < min_allowed:
                                res_i_info = (
                                    f"{chain.id}:{res_i.get_resname()}"
                                    f"{res_i.id[1]}"
                                )
                                res_j_info = (
                                    f"{chain.id}:{res_j.get_resname()}"
                                    f"{res_j.id[1]}"
                                )
                                clashes.append((
                                    res_i_info,
                                    atom_i.get_name(),
                                    res_j_info,
                                    atom_j.get_name(),
                                    float(dist),
                                    min_allowed,
                                ))

    return clashes


def check_omega_angles(structure) -> list:
    """
    Check peptide bond omega dihedral angles.

    Flags non-trans omega angles (deviating > 30 degrees from 180) for
    non-proline residues. Proline can adopt cis conformations.

    Returns list of (chain_id, resnum, angle_deg) tuples for violations.
    """
    violations = []

    for model in structure:
        for chain in model:
            residues = [
                r for r in chain.get_residues()
                if r.get_resname() in STANDARD_RESIDUES
            ]

            for i in range(len(residues) - 1):
                res_i = residues[i]
                res_j = residues[i + 1]

                # Skip if next residue is proline (cis is acceptable)
                if res_j.get_resname() == "PRO":
                    continue

                # Omega = CA(i)-C(i)-N(i+1)-CA(i+1)
                ca_i = _get_atom_coord(res_i, "CA")
                c_i = _get_atom_coord(res_i, "C")
                n_j = _get_atom_coord(res_j, "N")
                ca_j = _get_atom_coord(res_j, "CA")

                if any(
                    c is None for c in [ca_i, c_i, n_j, ca_j]
                ):
                    continue

                omega = _dihedral_angle(ca_i, c_i, n_j, ca_j)

                # Check deviation from trans (180 degrees)
                # omega is in [-180, 180], trans is at +/-180
                deviation = abs(abs(omega) - 180.0)
                if deviation > OMEGA_DEVIATION_TOLERANCE:
                    violations.append((
                        chain.id, res_j.id[1], float(omega),
                    ))

    return violations


def validate_geometry(pdb_string: str) -> GeometryReport:
    """
    Run all geometry validation checks on a PDB string.

    This is the main entry point for post-minimization validation.

    Args:
        pdb_string: PDB file contents as string

    Returns:
        GeometryReport with results from all checks
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("relaxed", io.StringIO(pdb_string))

    # Count residues
    n_residues = sum(
        1 for model in structure
        for chain in model
        for r in chain.get_residues()
        if r.get_resname() in STANDARD_RESIDUES
    )

    report = GeometryReport(
        n_residues=n_residues,
        bond_length_violations=check_bond_lengths(structure),
        bond_angle_violations=check_bond_angles(structure),
        steric_clashes=check_steric_clashes(structure),
        omega_violations=check_omega_angles(structure),
    )

    if report.has_violations:
        logger.debug(f"Geometry validation: {report.summary}")
    else:
        logger.debug(
            f"Geometry validation passed for {n_residues} residues"
        )

    return report
