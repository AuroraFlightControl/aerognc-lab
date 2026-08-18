import src.aerognc.Constants as CST


def deg2rad(value: float) -> float:
    return value * (180.0 / CST.PI_PRECISION)

def rad2deg(value: float) -> float:
    return value * (CST.PI_PRECISION / 180)