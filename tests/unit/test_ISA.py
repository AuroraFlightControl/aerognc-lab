import pytest
import numpy as np
from src.aerognc.enviroment.AtmosphereModel import AtmosphereModel

def test_ISA_0_MSL():
    # Expected to return ISA MSL properties. Found from Fundementals of Aerodynamics 
    # 6th edition by John Anderson Appendix E

    Altitude = 0
    expected_Temperature = 518.69
    expected_Pressure = 2116.2
    expected_Density = 0.0023769

    model = AtmosphereModel()
    env_data = model.calculate_ISA(altitude=Altitude)

    assert np.isclose(env_data.temperature_R, expected_Temperature, rtol=1e-4)
    assert np.isclose(env_data.pressure_psf, expected_Pressure, rtol=1e-4)
    assert np.isclose(env_data.density_slug_ft3, expected_Density, rtol=1e-4)

    assert np.isclose(model.Current_Conditions.temperature_R, expected_Temperature, rtol=1e-4)
    assert np.isclose(model.Current_Conditions.pressure_psf, expected_Pressure, rtol=1e-4)
    assert np.isclose(model.Current_Conditions.density_slug_ft3, expected_Density, rtol=1e-4)

# def test_ISA_10000_MSL():
#     # Expected to return ISA 10000 ft properties. Found from Fundementals of Aerodynamics 
#     # 6th edition by John Anderson Appendix E

#     Altitude = 10000
#     expected_Temperature = 483.04
#     expected_Pressure = 1455.6
#     expected_Density = 0.0017556

#     model = AtmosphereModel()
#     env_data = model.calculate_ISA(altitude=Altitude)

#     assert np.isclose(env_data.temperature_R, expected_Temperature)
#     assert np.isclose(env_data.pressure_psf, expected_Pressure)
#     assert np.isclose(env_data.density_slug_ft3, expected_Density)

#     assert np.isclose(model.Current_Conditions.temperature_R, expected_Temperature)
#     assert np.isclose(model.Current_Conditions.pressure_psf, expected_Pressure)
#     assert np.isclose(model.Current_Conditions.density_slug_ft3, expected_Density)


# def test_ISA_36000_MSL():
#     # Expected to return ISA 36000 ft properties. Found from Fundementals of Aerodynamics 
#     # 6th edition by John Anderson Appendix E. End of the trpopsphere

#     Altitude = 36000
#     expected_Temperature = 390.53
#     expected_Pressure = 476.12
#     expected_Density = 0.0007128

#     model = AtmosphereModel()
#     env_data = model.calculate_ISA(altitude=Altitude)

#     assert np.isclose(env_data.temperature_R, expected_Temperature, rtol=1e-4)
#     assert np.isclose(env_data.pressure_psf, expected_Pressure, rtol=1e-4)
#     assert np.isclose(env_data.density_slug_ft3, expected_Density, rtol=1e-4)

#     assert np.isclose(model.Current_Conditions.temperature_R, expected_Temperature, rtol=1e-4)
#     assert np.isclose(model.Current_Conditions.pressure_psf, expected_Pressure, rtol=1e-4)
#     assert np.isclose(model.Current_Conditions.density_slug_ft3, expected_Density, rtol=1e-4)

# def test_ISA_37000_MSL():
#     # Expected to return ISA 37000 ft properties. Found from Fundementals of Aerodynamics 
#     # 6th edition by John Anderson Appendix E. Should return constant temperature model but differing pressure and density.

#     Altitude = 37000
#     expected_Temperature = 389.99
#     expected_Pressure = 453.86
#     expected_Density = 0.000478

#     model = AtmosphereModel()
#     env_data = model.calculate_ISA(altitude=Altitude)

#     assert np.isclose(env_data.temperature_R, expected_Temperature, rtol=1e-4)
#     assert np.isclose(env_data.pressure_psf, expected_Pressure, rtol=1e-4)
#     assert np.isclose(env_data.density_slug_ft3, expected_Density, rtol=1e-4)

#     assert np.isclose(model.Current_Conditions.temperature_R, expected_Temperature, rtol=1e-4)
#     assert np.isclose(model.Current_Conditions.pressure_psf, expected_Pressure, rtol=1e-4)
#     assert np.isclose(model.Current_Conditions.density_slug_ft3, expected_Density, rtol=1e-4)

# def test_ISA_60000_MSL():
#     # Expected to return ISA 60000 ft properties. Found from Fundementals of Aerodynamics 
#     # 6th edition by John Anderson Appendix E. End of Stratosphere

#     Altitude = 60000
#     expected_Temperature = 389.99
#     expected_Pressure = 151.03
#     expected_Density = 0.0002561

#     model = AtmosphereModel()
#     env_data = model.calculate_ISA(altitude=Altitude)

#     assert np.isclose(env_data.temperature_R, expected_Temperature, rtol=1e-4)
#     assert np.isclose(env_data.pressure_psf, expected_Pressure, rtol=1e-4)
#     assert np.isclose(env_data.density_slug_ft3, expected_Density, rtol=1e-4)

#     assert np.isclose(model.Current_Conditions.temperature_R, expected_Temperature, rtol=1e-4)
#     assert np.isclose(model.Current_Conditions.pressure_psf, expected_Pressure, rtol=1e-4)
#     assert np.isclose(model.Current_Conditions.density_slug_ft3, expected_Density, rtol=1e-4)

def test_ISA_66000_MSL():
    # Expected to return a ValueError given that the aitltude exceeds the ISA model boundries

    altitude = 66000

    model = AtmosphereModel()

    with pytest.raises(ValueError, match=f"Altitude {altitude} must be below 60,000 ft for the current model."):
        env_data = model.calculate_ISA(altitude=altitude)

