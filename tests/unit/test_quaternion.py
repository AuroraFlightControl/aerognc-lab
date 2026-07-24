import os
import sys
import pytest
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.aerognc.math.quaternion import Quaternion, cross

def test_get_vector():
    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    vec = q.getVector()
    
    assert vec.shape == (4, 1)
    np.testing.assert_array_equal(vec, np.array([[1.0], [2.0], [3.0], [4.0]]))

def test_get_conjugate():
    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    conj = q.getConjugate()
    
    assert conj.q0 == 1.0
    assert conj.q1 == -2.0
    assert conj.q2 == -3.0
    assert conj.q3 == -4.0

def test_cross_product():
    # Test standard basis vectors: i x j = k
    i = np.array([[1.0], [0.0], [0.0]])
    j = np.array([[0.0], [1.0], [0.0]])
    expected_k = np.array([[0.0], [0.0], [1.0]])
    
    result = cross(i, j)
    np.testing.assert_array_equal(result, expected_k)

    # Test k x i = j
    k = np.array([[0.0], [0.0], [1.0]])
    expected_j = np.array([[0.0], [1.0], [0.0]])
    result2 = cross(k, i)
    np.testing.assert_array_equal(result2, expected_j)

def test_quaternion_multiply_identity():
    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    identity = Quaternion(1.0, 0.0, 0.0, 0.0)
    
    # q * 1 = q
    result = q1.multiply(identity)
    assert result == q1

def test_quaternion_multiply_basis_vectors():
    # Using fundamental quaternion units: i, j, k
    i = Quaternion(0.0, 1.0, 0.0, 0.0)
    j = Quaternion(0.0, 0.0, 1.0, 0.0)
    k = Quaternion(0.0, 0.0, 0.0, 1.0)
    
    # i * i = -1
    ii = i.multiply(i)
    assert ii == Quaternion(-1.0, 0.0, 0.0, 0.0)
    
    # i * j = k
    ij = i.multiply(j)
    assert ij == k
    
    # j * i = -k (non-commutative)
    ji = j.multiply(i)
    assert ji == Quaternion(0.0, 0.0, 0.0, -1.0)


def test_fromEuler_zero_rotation():
    # A 0-degree rotation in all axes should yield the identity quaternion [1, 0, 0, 0]
    q = Quaternion.fromEuler(0.0, 0.0, 0.0)
    
    assert q.q0 == pytest.approx(1.0)
    assert q.q1 == pytest.approx(0.0)
    assert q.q2 == pytest.approx(0.0)
    assert q.q3 == pytest.approx(0.0)

def test_fromEuler_pure_roll():
    # 90-degree (pi/2) roll around the X-axis
    q = Quaternion.fromEuler(math.pi / 2, 0.0, 0.0)
    
    expected = math.sqrt(2) / 2  # ~0.7071
    assert q.q0 == pytest.approx(expected)
    assert q.q1 == pytest.approx(expected)
    assert q.q2 == pytest.approx(0.0)
    assert q.q3 == pytest.approx(0.0)

def test_fromEuler_pure_pitch():
    # 90-degree (pi/2) pitch around the Y-axis
    q = Quaternion.fromEuler(0.0, math.pi / 2, 0.0)
    
    expected = math.sqrt(2) / 2
    assert q.q0 == pytest.approx(expected)
    assert q.q1 == pytest.approx(0.0)
    assert q.q2 == pytest.approx(expected)
    assert q.q3 == pytest.approx(0.0)

def test_fromEuler_pure_yaw():
    # 90-degree (pi/2) yaw around the Z-axis
    q = Quaternion.fromEuler(0.0, 0.0, math.pi / 2)
    
    expected = math.sqrt(2) / 2
    assert q.q0 == pytest.approx(expected)
    assert q.q1 == pytest.approx(0.0)
    assert q.q2 == pytest.approx(0.0)
    assert q.q3 == pytest.approx(expected)

def test_fromEuler_complex_rotation():
    # 90-degree roll and 90-degree pitch
    q = Quaternion.fromEuler(math.pi / 2, math.pi / 2, 0.0)
    
    # cos(45)*cos(45)*cos(0) + sin(45)*sin(45)*sin(0) = 0.5
    # sin(45)*cos(45)*cos(0) - cos(45)*sin(45)*sin(0) = 0.5
    # cos(45)*sin(45)*cos(0) + sin(45)*cos(45)*sin(0) = 0.5
    # cos(45)*cos(45)*sin(0) - sin(45)*sin(45)*cos(0) = -0.5
    
    assert q.q0 == pytest.approx(0.5)
    assert q.q1 == pytest.approx(0.5)
    assert q.q2 == pytest.approx(0.5)
    assert q.q3 == pytest.approx(-0.5)

    