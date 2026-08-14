import pytest
import math
from src.aerognc.math.ForwardEulerIntegrator import ForwardEulerIntegrator

# --- DUMMY CLASSES ---
class DummyDerivative:
    def __init__(self, values: float):
        self.values = values
        self.layout = "scalar_test"
        
class DummyState:
    def __init__(self, values: float):
        self.values = values
        self.layout = "scalar_test"
        
    def with_values(self, new_values: float):
        """Spawns a new state object with the updated values"""
        return DummyState(new_values)


# --- DERIVATIVE FUNCTION ---
def decay_derivative(t: float, state: DummyState) -> DummyDerivative:
    """A simple first-order derivative: dy/dt = -y"""
    return DummyDerivative(-state.values)


# --- TESTS ---
def test_forward_euler_accuracy():
    # Setup
    integrator = ForwardEulerIntegrator()
    
    # Initialize using our DummyState
    y = DummyState(1.0)
    t = 0.0
    dt = 0.001
    
    # Step forward 1 second
    for _ in range(1000):
        y = integrator.step(decay_derivative, t, y, dt)  # type: ignore[arg-type]
        t += dt
        
    # Extract the internal float value for the assertion
    exact_solution = math.exp(-1.0)
    assert math.isclose(y.values, exact_solution, rel_tol=1e-3)