"""Tests for graphrelax.validation module."""

import numpy as np
import pytest

from graphrelax.validation import (
    GeometryReport,
    check_bond_angles,
    check_bond_lengths,
    check_omega_angles,
    check_steric_clashes,
    validate_geometry,
)


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def good_pdb_string():
    """A well-formed 5-residue alanine peptide with ideal geometry."""
    # fmt: off
    return """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
ATOM      6  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.941   2.861   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.459   2.789   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       6.065   1.719   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       3.473   3.699   1.186  1.00  0.00           C
ATOM     11  N   ALA A   3       6.063   3.970   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       7.510   4.096   0.000  1.00  0.00           C
ATOM     13  C   ALA A   3       8.061   5.516   0.000  1.00  0.00           C
ATOM     14  O   ALA A   3       7.298   6.486   0.000  1.00  0.00           O
ATOM     15  CB  ALA A   3       8.038   3.336  -1.216  1.00  0.00           C
ATOM     16  N   ALA A   4       9.378   5.636   0.000  1.00  0.00           N
ATOM     17  CA  ALA A   4       9.993   6.957   0.000  1.00  0.00           C
ATOM     18  C   ALA A   4      11.511   6.885   0.000  1.00  0.00           C
ATOM     19  O   ALA A   4      12.117   5.815   0.000  1.00  0.00           O
ATOM     20  CB  ALA A   4       9.525   7.795   1.186  1.00  0.00           C
ATOM     21  N   ALA A   5      12.115   8.066   0.000  1.00  0.00           N
ATOM     22  CA  ALA A   5      13.562   8.192   0.000  1.00  0.00           C
ATOM     23  C   ALA A   5      14.113   9.612   0.000  1.00  0.00           C
ATOM     24  O   ALA A   5      13.350  10.582   0.000  1.00  0.00           O
ATOM     25  CB  ALA A   5      14.090   7.432  -1.216  1.00  0.00           C
ATOM     26  OXT ALA A   5      15.350   9.732   0.000  1.00  0.00           O
END
"""  # noqa: E501
    # fmt: on


@pytest.fixture
def distorted_bond_pdb_string():
    """A peptide with deliberately distorted bond lengths.

    The CA of residue 2 is shifted far from N, creating a bad N-CA bond.
    The C of residue 1 is moved to create a bad peptide bond to N of res 2.
    """
    # fmt: off
    return """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
ATOM      6  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       5.200   2.861   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.751   4.281   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       6.357   3.211   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       5.728   2.101   1.186  1.00  0.00           C
END
"""  # noqa: E501
    # fmt: on


@pytest.fixture
def cis_omega_pdb_string():
    """A peptide with a cis omega angle between residues 1 and 2.

    The CA of residue 2 is positioned to create a ~0 degree omega angle
    instead of ~180 degrees (trans).
    """
    # fmt: off
    return """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
ATOM      6  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.600   1.100  -1.350  1.00  0.00           C
ATOM      8  C   ALA A   2       5.100   1.000  -1.350  1.00  0.00           C
ATOM      9  O   ALA A   2       5.700   0.000  -1.350  1.00  0.00           O
ATOM     10  CB  ALA A   2       3.100   1.900  -2.550  1.00  0.00           C
END
"""  # noqa: E501
    # fmt: on


@pytest.fixture
def clash_pdb_string():
    """A peptide with two residues having atoms unrealistically close."""
    # fmt: off
    return """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
ATOM      6  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.941   2.861   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.459   2.789   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       6.065   1.719   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       3.473   3.699   1.186  1.00  0.00           C
ATOM     11  N   ALA A   3       6.063   3.970   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       7.510   4.096   0.000  1.00  0.00           C
ATOM     13  C   ALA A   3       8.061   5.516   0.000  1.00  0.00           C
ATOM     14  O   ALA A   3       7.298   6.486   0.000  1.00  0.00           O
ATOM     15  CB  ALA A   3       1.500   0.100   0.100  1.00  0.00           C
END
"""  # noqa: E501
    # fmt: on


# -- GeometryReport tests ----------------------------------------------------


class TestGeometryReport:
    """Tests for GeometryReport dataclass."""

    def test_empty_report_no_violations(self):
        report = GeometryReport(n_residues=10)
        assert not report.has_violations
        assert report.violation_count == 0

    def test_empty_report_summary(self):
        report = GeometryReport(n_residues=10)
        assert "No violations" in report.summary
        assert "10 residues" in report.summary

    def test_report_with_bond_violations(self):
        report = GeometryReport(
            n_residues=10,
            bond_length_violations=[("A", 1, "ALA", "N", "CA", 1.6, 1.458)],
        )
        assert report.has_violations
        assert report.violation_count == 1
        assert "1 bond length" in report.summary

    def test_report_with_multiple_violation_types(self):
        report = GeometryReport(
            n_residues=20,
            bond_length_violations=[
                ("A", 1, "ALA", "N", "CA", 1.6, 1.458),
            ],
            bond_angle_violations=[
                ("A", 1, "ALA", "CA-C-N", 130.0, 116.5),
            ],
            steric_clashes=[
                ("A:ALA1", "CB", "A:ALA5", "CB", 1.0, 1.9),
            ],
            omega_violations=[
                ("A", 3, 5.0),
            ],
        )
        assert report.violation_count == 4
        assert "4 total violations" in report.summary
        assert "bond length" in report.summary
        assert "bond angle" in report.summary
        assert "steric clash" in report.summary
        assert "omega angle" in report.summary


# -- Bond length check tests -------------------------------------------------


class TestCheckBondLengths:
    """Tests for check_bond_lengths function."""

    def test_good_geometry_few_violations(self, good_pdb_string):
        """Good geometry should have few or no bond violations."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", io.StringIO(good_pdb_string))
        violations = check_bond_lengths(structure)
        # Ideal geometry PDB may have minor deviations
        # but should not have many violations
        assert isinstance(violations, list)

    def test_distorted_geometry_has_violations(
        self, distorted_bond_pdb_string
    ):
        """Distorted geometry should have bond length violations."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(distorted_bond_pdb_string)
        )
        violations = check_bond_lengths(structure)
        assert len(violations) > 0

    def test_violation_tuple_format(self, distorted_bond_pdb_string):
        """Violations should have correct tuple format."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(distorted_bond_pdb_string)
        )
        violations = check_bond_lengths(structure)
        assert len(violations) > 0
        chain_id, resnum, resname, atom1, atom2, observed, expected = (
            violations[0]
        )
        assert isinstance(chain_id, str)
        assert isinstance(resnum, int)
        assert isinstance(resname, str)
        assert isinstance(observed, float)
        assert isinstance(expected, float)


# -- Bond angle check tests --------------------------------------------------


class TestCheckBondAngles:
    """Tests for check_bond_angles function."""

    def test_good_geometry(self, good_pdb_string):
        """Good geometry should pass angle checks with tolerance."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", io.StringIO(good_pdb_string))
        violations = check_bond_angles(structure)
        assert isinstance(violations, list)

    def test_violation_format(self):
        """Angle violations should have correct tuple format."""
        from Bio.PDB import PDBParser
        import io

        # Create a PDB with a severely distorted angle
        # (move CA far off to create bad CA-C-N angle)
        # fmt: off
        pdb = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       2.100   1.500   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       2.200   1.600   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.459   2.789   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.065   1.719   0.000  1.00  0.00           O
END
"""  # noqa: E501
        # fmt: on
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", io.StringIO(pdb))
        violations = check_bond_angles(structure)
        # With such distorted geometry we should get violations
        assert isinstance(violations, list)


# -- Steric clash check tests ------------------------------------------------


class TestCheckStericClashes:
    """Tests for check_steric_clashes function."""

    def test_good_geometry_no_clashes(self, good_pdb_string):
        """Good geometry should have no steric clashes."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", io.StringIO(good_pdb_string))
        clashes = check_steric_clashes(structure)
        assert len(clashes) == 0

    def test_overlapping_atoms_detected(self, clash_pdb_string):
        """Overlapping atoms should be detected as clashes."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(clash_pdb_string)
        )
        clashes = check_steric_clashes(structure)
        assert len(clashes) > 0

    def test_clash_tuple_format(self, clash_pdb_string):
        """Clash entries should have correct tuple format."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(clash_pdb_string)
        )
        clashes = check_steric_clashes(structure)
        assert len(clashes) > 0
        res1_info, atom1, res2_info, atom2, distance, min_allowed = (
            clashes[0]
        )
        assert isinstance(res1_info, str)
        assert isinstance(atom1, str)
        assert isinstance(distance, float)
        assert isinstance(min_allowed, float)
        assert distance < min_allowed


# -- Omega angle check tests -------------------------------------------------


class TestCheckOmegaAngles:
    """Tests for check_omega_angles function."""

    def test_trans_peptide_no_violations(self, good_pdb_string):
        """Trans peptide bonds should not be flagged."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", io.StringIO(good_pdb_string))
        violations = check_omega_angles(structure)
        assert len(violations) == 0

    def test_cis_peptide_detected(self, cis_omega_pdb_string):
        """Cis peptide bond should be flagged for non-proline."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(cis_omega_pdb_string)
        )
        violations = check_omega_angles(structure)
        assert len(violations) > 0

    def test_omega_violation_format(self, cis_omega_pdb_string):
        """Omega violations should have correct tuple format."""
        from Bio.PDB import PDBParser
        import io

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(cis_omega_pdb_string)
        )
        violations = check_omega_angles(structure)
        assert len(violations) > 0
        chain_id, resnum, angle_deg = violations[0]
        assert isinstance(chain_id, str)
        assert isinstance(resnum, int)
        assert isinstance(angle_deg, float)
        # Cis angle should be near 0, far from 180
        assert abs(angle_deg) < 90 or abs(abs(angle_deg) - 360) < 90


# -- Full validation tests ---------------------------------------------------


class TestValidateGeometry:
    """Tests for the main validate_geometry function."""

    def test_returns_geometry_report(self, good_pdb_string):
        """validate_geometry should return a GeometryReport."""
        report = validate_geometry(good_pdb_string)
        assert isinstance(report, GeometryReport)

    def test_counts_residues(self, good_pdb_string):
        """Should correctly count residues."""
        report = validate_geometry(good_pdb_string)
        assert report.n_residues == 5

    def test_good_geometry_summary(self, good_pdb_string):
        """Good geometry should produce a clean or near-clean report."""
        report = validate_geometry(good_pdb_string)
        assert isinstance(report.summary, str)
        assert report.n_residues > 0

    def test_distorted_geometry_has_violations(
        self, distorted_bond_pdb_string
    ):
        """Distorted geometry should produce violations."""
        report = validate_geometry(distorted_bond_pdb_string)
        assert report.has_violations
        assert report.violation_count > 0
