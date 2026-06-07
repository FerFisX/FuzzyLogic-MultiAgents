"""
tests/test_fie.py  ─  FIE unit & integration tests
====================================================
Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import numpy as np

from fie.engine import (
    _mu_tri, _mu_trap, _mu,
    fuzzify, evaluate, evaluar_propuesta,
    FIEResult, IEN_ESCALATION_THRESHOLD,
)


# ===========================================================================
# MEMBERSHIP FUNCTION TESTS
# ===========================================================================
class TestMembershipFunctions:

    def test_triangular_peak(self):
        """Peak of triangular MF must equal 1."""
        assert _mu_tri(50.0, 30, 50, 70) == pytest.approx(1.0)

    def test_triangular_left_foot(self):
        assert _mu_tri(30.0, 30, 50, 70) == pytest.approx(0.0, abs=1e-6)

    def test_triangular_right_foot(self):
        assert _mu_tri(70.0, 30, 50, 70) == pytest.approx(0.0, abs=1e-6)

    def test_triangular_midpoint_rising(self):
        assert _mu_tri(40.0, 30, 50, 70) == pytest.approx(0.5)

    def test_trapezoidal_plateau(self):
        """Plateau region must equal 1."""
        for x in [20, 30, 40, 50]:
            assert _mu_trap(x, 10, 20, 50, 60) == pytest.approx(1.0)

    def test_trapezoidal_zero_outside(self):
        assert _mu_trap(5.0,  10, 20, 50, 60) == pytest.approx(0.0, abs=1e-6)
        assert _mu_trap(65.0, 10, 20, 50, 60) == pytest.approx(0.0, abs=1e-6)

    def test_trapezoidal_left_slope(self):
        assert _mu_trap(15.0, 10, 20, 50, 60) == pytest.approx(0.5)

    def test_trapezoidal_right_slope(self):
        assert _mu_trap(55.0, 10, 20, 50, 60) == pytest.approx(0.5)

    def test_vectorised_returns_array(self):
        x = np.array([0, 25, 50, 75, 100], dtype=float)
        y = _mu_trap(x, 0, 0, 25, 45)
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape

    def test_boundary_values_clamp(self):
        """Values outside universe should return 0, not raise."""
        assert _mu_tri(-10.0, 30, 50, 70) == pytest.approx(0.0, abs=1e-6)
        assert _mu_tri(110.0, 30, 50, 70) == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# FUZZIFICATION TESTS
# ===========================================================================
class TestFuzzify:

    def test_returns_all_variables(self):
        d = fuzzify(50.0, 0.5, 75.0)
        assert set(d.keys()) == {"cs", "ir", "fh"}

    def test_membership_degrees_in_range(self):
        d = fuzzify(50.0, 0.5, 75.0)
        for var, terms in d.items():
            for label, mu in terms.items():
                assert 0.0 <= mu <= 1.0, f"{var}.{label} = {mu}"

    def test_extreme_low_cs(self):
        d = fuzzify(0.0, 0.0, 100.0)
        assert d["cs"]["baja"] == pytest.approx(1.0)
        assert d["cs"]["alta"] == pytest.approx(0.0, abs=1e-6)

    def test_extreme_high_cs(self):
        d = fuzzify(100.0, 1.0, 0.0)
        assert d["cs"]["alta"] == pytest.approx(1.0)
        assert d["cs"]["baja"] == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# FIE ENGINE — PAPER TEST CASES
# ===========================================================================
class TestEvaluatePaperCases:
    """
    Mirror the two assertions from fuzzy_inference_engine.py Test A & B,
    plus additional boundary and routing tests.
    """

    # Test A (from original file): high complexity, high uncertainty, poor reliability
    def test_case_A_low_confidence_critical_escalation(self):
        nc, ien = evaluar_propuesta(cs=70.0, ir=0.8, fh=40.0)
        assert nc  < 0.4, f"Expected low NC, got {nc:.4f}"
        assert ien > 70.0, f"Expected high IEN, got {ien:.4f}"

    # Test B (from original file): easy task, certain, excellent agent
    def test_case_B_high_confidence_local(self):
        nc, ien = evaluar_propuesta(cs=20.0, ir=0.1, fh=95.0)
        assert nc  > 0.8,  f"Expected high NC, got {nc:.4f}"
        assert ien < 30.0, f"Expected low IEN, got {ien:.4f}"

    # From motor_difuso.py Case 1
    def test_caso1_easy_secure(self):
        nc, ien = evaluar_propuesta(cs=10, ir=0.1, fh=95)
        assert nc > 0.75, f"Got {nc:.4f}"
        assert ien < 35.0, f"Got {ien:.4f}"

    # From motor_difuso.py Case 2
    def test_caso2_hard_doubtful(self):
        nc, ien = evaluar_propuesta(cs=85, ir=0.6, fh=70)
        assert nc < 0.65, f"Got {nc:.4f}"

    def test_escalation_flag_set_when_ien_high(self):
        result = evaluate(cs=90, ir=0.9, fh=10)
        assert result.should_escalate is True
        assert result.ien >= IEN_ESCALATION_THRESHOLD

    def test_no_escalation_flag_when_ien_low(self):
        result = evaluate(cs=5, ir=0.05, fh=98)
        assert result.should_escalate is False
        assert result.ien < IEN_ESCALATION_THRESHOLD

    def test_result_has_diagnostics(self):
        result = evaluate(cs=50, ir=0.5, fh=70)
        assert isinstance(result.nc_activations, dict)
        assert isinstance(result.ien_activations, dict)
        assert result.eval_ms >= 0.0

    def test_eval_ms_under_10ms(self):
        """FIE must complete in under 10 ms on commodity hardware."""
        result = evaluate(cs=50, ir=0.5, fh=70)
        assert result.eval_ms < 10.0, f"Too slow: {result.eval_ms:.2f} ms"


# ===========================================================================
# INPUT VALIDATION
# ===========================================================================
class TestInputValidation:

    def test_cs_out_of_range_high(self):
        with pytest.raises(ValueError):
            evaluate(cs=101, ir=0.5, fh=50)

    def test_cs_out_of_range_low(self):
        with pytest.raises(ValueError):
            evaluate(cs=-1, ir=0.5, fh=50)

    def test_ir_out_of_range(self):
        with pytest.raises(ValueError):
            evaluate(cs=50, ir=1.5, fh=50)

    def test_fh_out_of_range(self):
        with pytest.raises(ValueError):
            evaluate(cs=50, ir=0.5, fh=-5)

    def test_boundary_values_accepted(self):
        """Exact boundary values must not raise."""
        evaluate(cs=0.0,   ir=0.0, fh=0.0)
        evaluate(cs=100.0, ir=1.0, fh=100.0)


# ===========================================================================
# DEFUZZIFICATION STABILITY
# ===========================================================================
class TestDefuzzStability:

    def test_output_always_in_range(self):
        """Sweep over grid of inputs and check output bounds."""
        for cs in [0, 25, 50, 75, 100]:
            for ir in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for fh in [0, 25, 50, 75, 100]:
                    result = evaluate(cs, ir, fh)
                    assert 0.0 <= result.nc  <= 1.0,   f"NC out of range: {result.nc}"
                    assert 0.0 <= result.ien <= 100.0, f"IEN out of range: {result.ien}"

    def test_monotonicity_nc_vs_fh(self):
        """Higher FH should generally increase NC (holding CS, IR constant)."""
        _, nc_low  = evaluate(cs=50, ir=0.4, fh=20).nc, evaluate(cs=50, ir=0.4, fh=20).nc
        nc_low  = evaluate(cs=50, ir=0.4, fh=20).nc
        nc_high = evaluate(cs=50, ir=0.4, fh=90).nc
        assert nc_high > nc_low, f"Expected NC(FH=90) > NC(FH=20): {nc_high:.3f} vs {nc_low:.3f}"

    def test_monotonicity_ien_vs_ir(self):
        """Higher IR should generally increase IEN."""
        ien_low  = evaluate(cs=60, ir=0.1, fh=70).ien
        ien_high = evaluate(cs=60, ir=0.9, fh=70).ien
        assert ien_high > ien_low


# ===========================================================================
# COMPATIBILITY ALIAS
# ===========================================================================
class TestCompatibilityAlias:

    def test_evaluar_propuesta_returns_tuple(self):
        result = evaluar_propuesta(cs=40, ir=0.3, fh=80)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_evaluar_propuesta_matches_evaluate(self):
        nc1, ien1 = evaluar_propuesta(cs=55, ir=0.45, fh=65)
        r = evaluate(cs=55, ir=0.45, fh=65)
        assert nc1  == pytest.approx(r.nc,  abs=1e-6)
        assert ien1 == pytest.approx(r.ien, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
