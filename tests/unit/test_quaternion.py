import pytest
from src.aerognc.math.quaternion import Quat
import src.aerognc.Constants as CST
import math

def test_initialization():
    q = Quat(1.0, 0.0, 0.0, 0.0)
    assert q.w == 1.0
    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == 0.0

def test_normalization():
    with pytest.raises(ValueError):
        q = Quat(2.0, 0.0, 0.0, 0.0)  # Not normalized

    with pytest.raises(ValueError):
        q = Quat(1.0, 1.0, 1.0, 1.0)  # Not normalized

def test_valid_quaternion():
    q = Quat(0.7071068, 0.7071068, 0.0, 0.0)  # Normalized quaternion
    assert q.w == pytest.approx(0.7071068, rel=1e-9)
    assert q.x == pytest.approx(0.7071068, rel=1e-9)
    assert q.y == pytest.approx(0.0, rel=1e-9)
    assert q.z == pytest.approx(0.0, rel=1e-9)

def test_from_axis_angle():
    angle = CST.PI_PRECISION / 2 # 90 degrees
    q = Quat.from_axis_angle(1.0, 0.0, 0.0, angle)
    tol = 1e-6
    assert q.w == pytest.approx(0.7071068, rel=tol)
    assert q.x == pytest.approx(0.7071068, rel=tol)
    assert q.y == pytest.approx(0.0, rel=tol)
    assert q.z == pytest.approx(0.0, rel=tol)

def test_quaternion_multiplication():
    # Define our base unit quaternions (all have a norm of 1.0)
    q_identity = Quat(1.0, 0.0, 0.0, 0.0) # 1
    q_i = Quat(0.0, 1.0, 0.0, 0.0)        # i
    q_j = Quat(0.0, 0.0, 1.0, 0.0)        # j
    q_k = Quat(0.0, 0.0, 0.0, 1.0)        # k

    # Test 1: Identity (1 * i = i)
    result_identity = q_identity.multiply(q_i) 
    assert math.isclose(result_identity.w, 0.0, abs_tol=1e-9)
    assert math.isclose(result_identity.x, 1.0, abs_tol=1e-9)
    assert math.isclose(result_identity.y, 0.0, abs_tol=1e-9)
    assert math.isclose(result_identity.z, 0.0, abs_tol=1e-9)

    # Test 2: Standard multiplication (i * j = k)
    result_ij = q_i.multiply(q_j)
    assert math.isclose(result_ij.w, 0.0, abs_tol=1e-9)
    assert math.isclose(result_ij.x, 0.0, abs_tol=1e-9)
    assert math.isclose(result_ij.y, 0.0, abs_tol=1e-9)
    assert math.isclose(result_ij.z, 1.0, abs_tol=1e-9)

    # Test 3: Non-commutative property (j * i = -k)
    result_ji = q_j.multiply(q_i)
    assert math.isclose(result_ji.w, 0.0, abs_tol=1e-9)
    assert math.isclose(result_ji.x, 0.0, abs_tol=1e-9)
    assert math.isclose(result_ji.y, 0.0, abs_tol=1e-9)
    assert math.isclose(result_ji.z, -1.0, abs_tol=1e-9)

    # Test 4: Squared imaginary units (i * i = -1)
    result_ii = q_i.multiply(q_i)
    assert math.isclose(result_ii.w, -1.0, abs_tol=1e-9)
    assert math.isclose(result_ii.x, 0.0, abs_tol=1e-9)
    assert math.isclose(result_ii.y, 0.0, abs_tol=1e-9)
    assert math.isclose(result_ii.z, 0.0, abs_tol=1e-9)


def test_quaternion_derivative():
    # Setup: Base orientation is Identity (no rotation)
    q_identity = Quat(1.0, 0.0, 0.0, 0.0)

    # Test 1: Rotation around X-axis
    # Angular velocity of 2 rad/s around the X axis
    # Expected derivative: 0.5 * [1,0,0,0] * [0,2,0,0] = [0, 1, 0, 0]
    dw, dx, dy, dz = q_identity.derivative(2.0, 0.0, 0.0)
    
    assert math.isclose(dw, 0.0, abs_tol=1e-9)
    assert math.isclose(dx, 1.0, abs_tol=1e-9)
    assert math.isclose(dy, 0.0, abs_tol=1e-9)
    assert math.isclose(dz, 0.0, abs_tol=1e-9)

    # Test 2: Rotation from a 90-degree pitch (j)
    q_pitch = Quat(0.0, 0.0, 1.0, 0.0) 
    
    # Angular velocity of 2 rad/s around the Y axis
    # Expected derivative: 0.5 * [0,0,1,0] * [0,0,2,0] = [-1, 0, 0, 0]
    dw2, dx2, dy2, dz2 = q_pitch.derivative(0.0, 2.0, 0.0)
    
    assert math.isclose(dw2, -1.0, abs_tol=1e-9)
    assert math.isclose(dx2, 0.0, abs_tol=1e-9)
    assert math.isclose(dy2, 0.0, abs_tol=1e-9)
    assert math.isclose(dz2, 0.0, abs_tol=1e-9)


def test_fromEuler_zero_rotation():
    # A 0-degree rotation in all axes should yield the identity quaternion [1, 0, 0, 0]
    q = Quat.from_euler_angles(0.0, 0.0, 0.0)
    
    
    assert q.w == pytest.approx(1.0)
    assert q.x == pytest.approx(0.0)
    assert q.y == pytest.approx(0.0)
    assert q.z == pytest.approx(0.0)

def test_fromEuler_pure_roll():
    # 90-degree (pi/2) roll around the X-axis
    q = Quat.from_euler_angles(math.pi / 2, 0.0, 0.0)
    
    expected = math.sqrt(2) / 2  # ~0.7071
    assert q.w == pytest.approx(expected)
    assert q.x == pytest.approx(expected)
    assert q.y == pytest.approx(0.0)
    assert q.z == pytest.approx(0.0)

def test_fromEuler_pure_pitch():
    # 90-degree (pi/2) pitch around the Y-axis
    q = Quat.from_euler_angles(0.0, math.pi / 2, 0.0)
    
    expected = math.sqrt(2) / 2
    assert q.w == pytest.approx(expected)
    assert q.x == pytest.approx(0.0)
    assert q.y == pytest.approx(expected)
    assert q.z == pytest.approx(0.0)

def test_fromEuler_pure_yaw():
    # 90-degree (pi/2) yaw around the Z-axis
    q = Quat.from_euler_angles(0.0, 0.0, math.pi / 2)
    
    expected = math.sqrt(2) / 2
    assert q.w == pytest.approx(expected)
    assert q.x == pytest.approx(0.0)
    assert q.y == pytest.approx(0.0)
    assert q.z == pytest.approx(expected)

def test_fromEuler_complex_rotation():
    # 90-degree roll and 90-degree pitch
    q = Quat.from_euler_angles(math.pi / 2, math.pi / 2, 0.0)
    
    # cos(45)*cos(45)*cos(0) + sin(45)*sin(45)*sin(0) = 0.5
    # sin(45)*cos(45)*cos(0) - cos(45)*sin(45)*sin(0) = 0.5
    # cos(45)*sin(45)*cos(0) + sin(45)*cos(45)*sin(0) = 0.5
    # cos(45)*cos(45)*sin(0) - sin(45)*sin(45)*cos(0) = -0.5
    
    assert q.w == pytest.approx(0.5)
    assert q.x == pytest.approx(0.5)
    assert q.y == pytest.approx(0.5)
    assert q.z == pytest.approx(-0.5)

def test_conjugate_and_inverse():
    # Setup a valid quaternion (60-degree rotation around an arbitrary axis)
    q = Quat(0.8660254, 0.2886751, 0.2886751, 0.2886751)
    
    q_inv = q.inverse()
    assert q_inv.w == 0.8660254
    assert q_inv.x == -0.2886751
    assert q_inv.y == -0.2886751
    assert q_inv.z == -0.2886751

    # Multiplying a quaternion by its inverse MUST yield the identity quaternion
    identity = q.multiply(q_inv)
    assert identity.w == pytest.approx(1.0, abs=1e-6)
    assert identity.x == pytest.approx(0.0, abs=1e-6)
    assert identity.y == pytest.approx(0.0, abs=1e-6)
    assert identity.z == pytest.approx(0.0, abs=1e-6)

def test_rotate_vector():
    # Create a 90-degree yaw rotation (around Z-axis)
    q = Quat.from_euler_angles(0.0, 0.0, math.pi / 2)
    
    # Rotate the X-unit vector (1, 0, 0)
    # Expected: The X-axis rotates to become the Y-axis (0, 1, 0)
    rx, ry, rz = q.rotate_vector(1.0, 0.0, 0.0)
    
    assert rx == pytest.approx(0.0, abs=1e-9)
    assert ry == pytest.approx(1.0, abs=1e-9)
    assert rz == pytest.approx(0.0, abs=1e-9)

def test_rotate_vector_complex():
    # 180-degree roll (around X-axis)
    q = Quat.from_euler_angles(math.pi, 0.0, 0.0)
    
    # Rotate the vector (0, 1, 1)
    # Expected: Y and Z components should invert -> (0, -1, -1)
    rx, ry, rz = q.rotate_vector(0.0, 1.0, 1.0)
    
    assert rx == pytest.approx(0.0, abs=1e-9)
    assert ry == pytest.approx(-1.0, abs=1e-9)
    assert rz == pytest.approx(-1.0, abs=1e-9)