"""
3PL IRT 模型测试
测试概率函数、信息函数、参数校准和向后兼容性
"""

import math
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from engine.scoring import (
    probability_3pl,
    calculate_new_theta,
    item_information,
    calibrate_item_parameters,
)


class TestProbability3PL:
    """P(θ) = c + (1 - c) / (1 + exp(-a * (θ - b)))"""

    def test_midpoint_property(self):
        """P(θ=b) should equal (1+c)/2 — the midpoint property of 3PL."""
        for c in [0.0, 0.2, 0.25]:
            p = probability_3pl(theta=1.0, b=1.0, a=1.5, c=c)
            expected = (1.0 + c) / 2.0
            assert abs(p - expected) < 1e-9, f"c={c}: P={p}, expected={expected}"

    def test_guessing_floor(self):
        """P(θ → -∞) should approach c (guessing parameter acts as floor)."""
        for c in [0.0, 0.2, 0.3]:
            p = probability_3pl(theta=-100.0, b=0.0, a=1.0, c=c)
            assert abs(p - c) < 1e-6, f"c={c}: P(-∞)={p}, expected≈{c}"

    def test_ceiling(self):
        """P(θ → +∞) should approach 1.0."""
        p = probability_3pl(theta=100.0, b=0.0, a=1.0, c=0.2)
        assert abs(p - 1.0) < 1e-6

    def test_monotonically_increasing(self):
        """P should increase with theta for fixed b, a, c."""
        prev = 0.0
        for theta in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            p = probability_3pl(theta, b=0.0, a=1.0, c=0.2)
            assert p >= prev
            prev = p


class TestDiscriminationEffect:
    """Higher discrimination → steeper curve → bigger theta changes."""

    def test_higher_a_steeper_curve(self):
        """With higher a, the probability difference between θ=b+1 and θ=b-1 is larger."""
        b = 0.0
        c = 0.2
        # Low discrimination
        p_high_low_a = probability_3pl(1.0, b, a=0.5, c=c)
        p_low_low_a = probability_3pl(-1.0, b, a=0.5, c=c)
        diff_low_a = p_high_low_a - p_low_low_a

        # High discrimination
        p_high_high_a = probability_3pl(1.0, b, a=2.5, c=c)
        p_low_high_a = probability_3pl(-1.0, b, a=2.5, c=c)
        diff_high_a = p_high_high_a - p_low_high_a

        assert diff_high_a > diff_low_a

    def test_higher_a_bigger_theta_update(self):
        """Higher discrimination should cause larger theta adjustments on correct answers."""
        theta = 0.0
        b = 0.0
        # Low a: expected probability is higher due to less steep curve at midpoint
        # Actually with 3PL, at theta=b, P = (1+c)/2 regardless of a.
        # So let's test at theta != b.
        theta = -0.5
        new_low_a = calculate_new_theta(theta, b, True, discrimination=0.5, guessing=0.2)
        new_high_a = calculate_new_theta(theta, b, True, discrimination=2.5, guessing=0.2)
        # Higher a means lower P at theta < b, so residual (1 - P) is larger → bigger update
        assert (new_high_a - theta) > (new_low_a - theta)


class TestInformationFunction:
    """I(θ) = a² * (P - c)² * (1 - P) / ((1 - c)² * P)"""

    def test_positive(self):
        """Information should be non-negative."""
        for theta in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            info = item_information(theta, b=0.0, a=1.0, c=0.2)
            assert info >= 0.0

    def test_peak_near_difficulty(self):
        """Information should peak near θ ≈ b (slightly above due to guessing)."""
        b = 1.0
        a = 1.5
        c = 0.2
        # Sample information at many points
        thetas = [b + d * 0.1 for d in range(-30, 31)]
        infos = [item_information(t, b, a, c) for t in thetas]
        peak_theta = thetas[infos.index(max(infos))]
        # Peak should be within 1.0 of b (shifted slightly above b due to c > 0)
        assert abs(peak_theta - b) < 1.0, f"Peak at {peak_theta}, expected near {b}"

    def test_higher_a_more_information(self):
        """Higher discrimination should yield more information at peak."""
        b = 0.0
        info_low = max(item_information(t, b, a=0.5, c=0.2) for t in [-3.0 + 0.1 * i for i in range(61)])
        info_high = max(item_information(t, b, a=2.5, c=0.2) for t in [-3.0 + 0.1 * i for i in range(61)])
        assert info_high > info_low

    def test_zero_at_extremes(self):
        """Information should approach 0 at extreme theta values."""
        info_low = item_information(-100.0, b=0.0, a=1.0, c=0.2)
        info_high = item_information(100.0, b=0.0, a=1.0, c=0.2)
        assert info_low < 0.001
        assert info_high < 0.001


class TestBackwardCompatibility:
    """Calling calculate_new_theta without discrimination/guessing should still work."""

    def test_three_arg_call(self):
        """Original 3-argument call should not raise."""
        result = calculate_new_theta(0.0, 0.0, True)
        assert isinstance(result, float)
        assert -3.0 <= result <= 3.0

    def test_correct_increases_theta(self):
        """Correct answer should increase theta."""
        new = calculate_new_theta(0.0, 0.0, True)
        assert new > 0.0

    def test_wrong_decreases_theta(self):
        """Wrong answer should decrease theta."""
        new = calculate_new_theta(0.0, 0.0, False)
        assert new < 0.0

    def test_gmat_score_range(self):
        """GMAT score should remain in valid range."""
        from engine.scoring import estimate_gmat_score
        for theta in [-3.0, 0.0, 3.0]:
            score = estimate_gmat_score(theta)
            assert 20 <= score <= 51

    def test_clamping(self):
        """Theta should be clamped to [-3, 3]."""
        new = calculate_new_theta(3.0, -3.0, True)
        assert new <= 3.0
        new = calculate_new_theta(-3.0, 3.0, False)
        assert new >= -3.0


class TestCalibrateItemParameters:
    """MLE calibration of a, b, c from synthetic response data."""

    def test_recovers_known_parameters(self):
        """With enough synthetic data, estimated params should be close to true values."""
        import random
        random.seed(42)

        true_a = 1.5
        true_b = 0.5
        true_c = 0.2

        # Generate 200 synthetic responses
        responses = []
        for _ in range(200):
            theta = random.uniform(-2.5, 2.5)
            p = probability_3pl(theta, true_b, true_a, true_c)
            is_correct = random.random() < p
            responses.append({"theta": theta, "is_correct": is_correct})

        result = calibrate_item_parameters(responses)

        assert result["converged"] is True
        assert abs(result["a"] - true_a) < 0.5, f"a={result['a']}, expected≈{true_a}"
        assert abs(result["b"] - true_b) < 0.5, f"b={result['b']}, expected≈{true_b}"
        assert abs(result["c"] - true_c) < 0.15, f"c={result['c']}, expected≈{true_c}"

    def test_insufficient_data_returns_defaults(self):
        """With < 5 responses, should return initial values and converged=False."""
        responses = [
            {"theta": 0.0, "is_correct": True},
            {"theta": 1.0, "is_correct": False},
        ]
        result = calibrate_item_parameters(responses)
        assert result["converged"] is False
        assert result["a"] == 1.0
        assert result["b"] == 0.0
        assert result["c"] == 0.2

    def test_return_structure(self):
        """Result should contain a, b, c, converged keys."""
        import random
        random.seed(99)
        responses = [{"theta": random.uniform(-2, 2), "is_correct": random.random() < 0.6} for _ in range(50)]
        result = calibrate_item_parameters(responses)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "converged" in result
        assert 0.5 <= result["a"] <= 2.5
        assert -3.0 <= result["b"] <= 3.0
        assert 0.0 <= result["c"] <= 0.35
