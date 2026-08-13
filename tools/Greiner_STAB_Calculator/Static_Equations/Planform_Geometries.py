import math
from typing import Optional
from dataclasses import dataclass

@dataclass
class WingGeo:
    cr:     float # [ft]
    ct:     float # [ft]
    b:      float # [ft]
    S:      float # [ft2]
    A:      float # [-]
    tr:     float # [-]
    MAC:    float # [ft]
    ymac:   float # [ft]
    xmac:   float # [ft]
    SWP_LE: float # [rad]
    SWP_25: float # [rad]
    SWP_50: float # [rad]
    SWP_75: float # [rad]
    SWP_1:  float # [rad]


def SLT_Taper(cr: float, ct: float) -> float:
    return ct/cr

def SLT_wingArea(b: float, cr: float, tr: float) -> float:
    """
    Function to calculate the wing area for a single, Linearly tapered wing planform
    Params:
        - b     -> Wing span [ft]
        - cr    -> Root Chord [ft]
        - tr    -> Taper Ratio [-]
    Output:
        - S     -> Wing Area [ft^2]
    """
    return (b/2) * cr * (1 + tr)

def SLT_MAC(cr: float, tr: float) -> float:
    """
    Function to calculate the Mean Aerodynamic Chord for a single, 
    Linearly tapered wing planform
    Params:
        - cr    -> Root Chord [ft]
        - tr    -> Taper Ratio [-]
    Output:
        - cb     -> Mean Aerodynamic Chord [ft]
    """
    return (2/3) * cr * ((1 + tr + tr**2)/(1 + tr))

def SLT_ymac(b: float, tr: float) -> float:
    return (b/6)*((1+2*tr)/(1+tr))

def AspectRatio(b: float, S: Optional[float] = None, c_avg: Optional[float] = None) -> float:

    if S is not None:
        return (b**2)/S
    elif c_avg is not None:
        return b/c_avg
    else:
        raise ValueError("Must include either wing area 'S' or average chord 'c_avg'")

def SLT_Cavg(cr: float, ct: float) -> float:
    return (cr + ct)/2

def SLT_xmac(ymac: float, SWP_LE: float) -> float:
    return ymac * math.tan(SWP_LE)

def SWP_n(n: float, m: float, SWP_m: float, tr: float, A: float) -> float:
    num = 4 * (n-m) * (1-tr)
    den = A * (1 + tr)
    tanSWP_m = math.tan(SWP_m)

    return math.atan(tanSWP_m - num/den)

def geo_twist(i_r: float, i_t: float) -> float:
    return i_t - i_r

def aero_twist(a_zlr: float, a_zlt: float, twist_geo: float) -> float:
    return twist_geo + a_zlr - a_zlt

def SLT_Wing(b: float, cr: float, ct: float, m: float, SWP_m: float) -> WingGeo:
    tr = SLT_Taper(cr, ct)
    S = SLT_wingArea(b, cr, tr)
    A = AspectRatio(b, S=S)
    MAC = SLT_MAC(cr, tr)
    ymac = SLT_ymac(b, tr)
    SWP_LE = SWP_n(0, m, SWP_m, tr, A)
    SWP_25 = SWP_n(0.25, m, SWP_m, tr, A)
    SWP_50 = SWP_n(0.50, m, SWP_m, tr, A)
    SWP_75 = SWP_n(0.75, m, SWP_m, tr, A)
    SWP_1 = SWP_n(1.0, m, SWP_m, tr, A)
    xmac = SLT_xmac(ymac, SWP_LE)

    return WingGeo(cr, ct, b, S, A, tr, MAC, ymac, xmac, SWP_LE, SWP_25, SWP_50, SWP_75, SWP_1)



def TEST_Wing():
    b = 6.00 # [ft]
    cr = 17.0 # [in]
    ct = 8.9 # [in]
    SWP_m = 0.0
    m = 0.25

    wing = SLT_Wing(b, cr/12, ct/12, m, SWP_m)

    print("==============================")
    print("Inputs:")
    print(f"    - Span:              {b} ft")
    print(f"    - Root Chord:        {cr/12} ft")
    print(f"    - Tip Chord:         {ct/12} ft")
    print(f"    - Sweep at {m*100}%:    {SWP_m} rad")
    print("==============================")
    print("Outputs:")
    print(f"    - Span:              {wing.b} ft")
    print(f"    - Root Chord:        {wing.cr} ft")
    print(f"    - Tip Chord:         {wing.ct} ft")
    print(f"    - Sweep at {m*100}%:    {SWP_m} rad")
    


if __name__=="__main__":
    TEST_Wing()