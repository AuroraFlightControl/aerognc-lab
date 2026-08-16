from src.aerognc.core.state import VehicleState



def check_ground_impact(time_s: float, state: VehicleState) -> str | None:
    # TODO: Current Implementation Assumes Ground at 0 ft MSL (No Terrain)

    position_ned_ft = state.section("position_ned_ft")

    if len(position_ned_ft) == 2:
        alt = -position_ned_ft[1]
    elif len(position_ned_ft) == 3:
        alt = -position_ned_ft[2]
    else:
        raise ValueError("Ground Collision Missing ALtitude from NED State.")


    if alt <= 0.0:
        return "Collision: Vehicle Hit the Ground"
    return None
