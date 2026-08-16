import pytest
import numpy as np

from src.aerognc.core.properties import MassProperties

def test_mass_properties_success():
    # Setup valid 3x3 inertia and 3-element CG
    valid_inertia = np.array([
        [100.0, 0.0, 0.0],
        [0.0, 200.0, 0.0],
        [0.0, 0.0, 300.0]
    ])
    valid_cg = np.array([1.5, 0.0, 0.5])
    
    props = MassProperties(
        mass_lbs=50.0,
        inertia_body_slug_ft2=valid_inertia,
        cg_body_ft=valid_cg
    )
    
    assert props.mass_lbs == 50.0
    assert np.array_equal(props.inertia_body_slug_ft2, valid_inertia)
    assert np.array_equal(props.cg_body_ft, valid_cg)

def test_mass_properties_invalid_mass():
    inertia = np.eye(3)
    cg = np.array([0.0, 0.0, 0.0])
    
    # Test zero mass
    with pytest.raises(ValueError, match="Mass must be positive."):
        MassProperties(mass_lbs=0.0, inertia_body_slug_ft2=inertia, cg_body_ft=cg)
        
    # Test negative mass
    with pytest.raises(ValueError, match="Mass must be positive."):
        MassProperties(mass_lbs=-10.0, inertia_body_slug_ft2=inertia, cg_body_ft=cg)

def test_mass_properties_invalid_inertia():
    # 1D array instead of 3x3
    invalid_inertia = np.array([1.0, 2.0, 3.0]) 
    cg = np.array([0.0, 0.0, 0.0])
    
    # We must escape the parentheses in the regex match string using backslashes
    with pytest.raises(ValueError, match=r"Inertia Matrix must have shape \(3, 3\)."):
        MassProperties(mass_lbs=10.0, inertia_body_slug_ft2=invalid_inertia, cg_body_ft=cg)

def test_mass_properties_invalid_cg():
    inertia = np.eye(3)
    # Shape (2,) instead of (3,)
    invalid_cg = np.array([1.0, 2.0]) 
    
    # Note: Matching the exact typo "shaper" from your source code!
    with pytest.raises(ValueError, match=r"Center of Gravity must have shaper \(3,\)."):
        MassProperties(mass_lbs=10.0, inertia_body_slug_ft2=inertia, cg_body_ft=invalid_cg)