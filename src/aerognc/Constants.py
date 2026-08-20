
PI_PRECISION = 3.1415962
ONE_G = 32.17405

# Sea Level ISA Properties

PRESSURE_MSL = 2116.2 
DENSITY_MSL = 0.0023769
TEMP_R_MSL = 518.69
TEMP_F_MSL = 59
DYNAMIC_VISC_MSL = 3.737e-7
SPEED_OF_SOUND_MSL = 1116.47
R_MSL = 1716.59



def deg2rad(value: float) -> float:
    return value * (PI_PRECISION / 180.0)

def rad2deg(value: float) -> float:
    return value * (180.0 / PI_PRECISION)