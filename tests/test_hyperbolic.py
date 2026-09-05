import numpy as np
import pytest

from phenotopo.hyperbolic import einstein_midpoint, place_patients, radial_specificity


def test_midpoint_of_symmetric_points_is_origin():
    pts = np.array([[0.5, 0.0], [-0.5, 0.0]])
    assert np.allclose(einstein_midpoint(pts), 0.0, atol=1e-9)


def test_midpoint_stays_inside_disk_and_between_points():
    pts = np.array([[0.9, 0.0], [0.0, 0.9]])
    m = einstein_midpoint(pts)
    assert np.linalg.norm(m) < 1.0
    assert m[0] > 0 and m[1] > 0


def test_weights_pull_midpoint_toward_heavier_point():
    pts = np.array([[0.8, 0.0], [-0.8, 0.0]])
    m = einstein_midpoint(pts, weights=[10.0, 1.0])
    assert m[0] > 0.3


def test_place_patients_ignores_unknown_terms_and_uses_weights():
    coords = {"a": np.array([0.6, 0.0]), "b": np.array([-0.6, 0.0])}
    out = place_patients([["a", "b"], ["a", "zzz"], []], coords, weights={"a": 5.0, "b": 1.0})
    assert out.shape == (3, 2)
    assert out[0, 0] > 0                       # pulled toward the heavier term
    assert np.allclose(out[1], coords["a"])    # unknown term ignored
    assert np.allclose(out[2], 0.0)            # no terms -> origin
    assert np.all(radial_specificity(out) <= 1.0)


def test_poincare_terms_requires_gensim_or_works():
    pytest.importorskip("gensim")
    from phenotopo.hyperbolic import poincare_terms
    rel = [("a", "root"), ("b", "root"), ("a1", "a"), ("a2", "a"), ("b1", "b")]
    coords = poincare_terms(rel, epochs=5, seed=0)
    assert set(coords) == {"root", "a", "b", "a1", "a2", "b1"}
    assert all(np.linalg.norm(v) < 1.0 for v in coords.values())
