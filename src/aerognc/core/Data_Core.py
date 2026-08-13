import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


# VehicleState
@dataclass
class VehicleState:
    state:      NDArray[np.float64]

# StateDerivative
@dataclass
class StateDerivitive:
    state:      NDArray[np.float64] 

# MassProperties
@dataclass
class MassProperties:
    mass:       float # lbs
    Ixx:        float
    Iyy:        float
    Izz:        float

# AtmosphereState
@dataclass
class AtmosphereState:
    temp:       float # deg F
    pressure:   float 
    dens:       float # slugs/ft^3


# WindState
@dataclass
class WindState:
    state:      NDArray[np.float64]

# AirData
@dataclass
class AirData:
    TAS:        float
    EAS:        float
    IAS:        float
    alpha:      float
    beta:       float

# ControlCommand

# ActuatorState

# ActuatorOutput

# PropulsionState

# PropulsionOutput

# ForceMoment

# SensorMeasurement

# EstimatedState

# OperatingPoint

# LinearModel

# SimulationResult