from src.aerognc.core.SimulationExec import SimulationResult
from src.aerognc.core.state import StateLayout
from src.aerognc.control.command_vector import CmdLayout
import matplotlib
matplotlib.use('Qt5Agg')  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_state_col_names(State_layout: StateLayout) -> list[str]:
    col_names = []

    sorted_slices = sorted(State_layout.slices.items(), key=lambda item: item[1].start)

    for name, slc in sorted_slices:
        width = slc.stop - slc.start

        if width == 1:
            col_names.append(f"state_{name}")
        else:
            for i in range(width):
                col_names.append(f"state_{name}_{i}")

    if len(col_names) != State_layout.size:
        raise ValueError(f"Incorrect coloumn generation, generated {len(col_names)} instead of {State_layout.size}.")

    return col_names

def get_cmd_col_names(Cmd_layout: CmdLayout) -> list[str]:
    col_names = []

    sorted_slices = sorted(Cmd_layout.slices.items(), key=lambda item: item[1].start)

    for name, slc in sorted_slices:
        width = slc.stop - slc.start

        if width == 1:
            col_names.append(f"Cmd_{name}")
        else:
            for i in range(width):
                col_names.append(f"Cmd_{name}_{i}")

    if len(col_names) != Cmd_layout.size:
        raise ValueError(f"Incorrect coloumn generation, generated {len(col_names)} instead of {Cmd_layout.size}.")

    return col_names

def orginize_telemetry(result: SimulationResult, state_layout: StateLayout, cmd_layout: CmdLayout) -> pd.DataFrame:

    state_names = get_state_col_names(State_layout=state_layout)
    cmd_names = get_cmd_col_names(Cmd_layout=cmd_layout)

    df_time = pd.DataFrame({"time_s": result.time })
    df_state = pd.DataFrame(result.state_values, columns=state_names)
    df_cmd = pd.DataFrame(result.command_values, columns=cmd_names)

    # Concatenate horizontally
    df = pd.concat([df_time, df_state, df_cmd], axis=1)

    # 3. Set time as the index for easier plotting
    df.set_index('time_s', inplace=True)

    return df


def visualize(result: SimulationResult, state_layout: StateLayout, cmd_layout: CmdLayout) -> None:

    df_telemetry = orginize_telemetry(
        result=result, 
        state_layout=state_layout, 
        cmd_layout=cmd_layout
        )

    print("\n\n Telemetry head:\n")
    print(df_telemetry.head(5))

    plt.figure(figsize=(10, 4))

    # Negate the 'Down' position to get standard Altitude
    downrange = df_telemetry['state_position_ned_ft_0']
    altitude = -df_telemetry['state_position_ned_ft_1'] 

    plt.plot(downrange, altitude, label="Flight Path")
    plt.title("2D Flight Trajectory")
    plt.xlabel("Downrange (North) [ft]")
    plt.ylabel("Altitude [ft]")
    plt.grid(True)
    plt.legend()
    #plt.show()



    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Convert to degrees
    theta_deg = df_telemetry['state_attitude_angle_rad'] * (180 / np.pi)
    q_deg_s = df_telemetry['state_body_rate_rad_s'] * (180 / np.pi)

    ax1.plot(df_telemetry.index, theta_deg, color='C1')
    ax1.set_ylabel("Pitch Angle, $\\theta$ [deg]")
    ax1.set_title("Pitch Dynamics over Time")
    ax1.grid(True)

    ax2.plot(df_telemetry.index, q_deg_s, color='C2')
    ax2.set_ylabel("Pitch Rate, $q$ [deg/s]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True)

    plt.tight_layout()
    #plt.show()

    # Derive Angle of Attack and Velocity
    u = df_telemetry['state_velocity_body_fps_0']
    w = df_telemetry['state_velocity_body_fps_1']

    df_telemetry['alpha_deg'] = np.arctan2(w, u) * (180 / np.pi)
    df_telemetry['total_airspeed_fps'] = np.sqrt(u**2 + w**2)

    # Plot them
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(df_telemetry.index, df_telemetry['alpha_deg'], color='purple')
    ax1.set_ylabel("Angle of Attack, $\\alpha$ [deg]")
    ax1.set_title("Aerodynamic States")
    ax1.grid(True)

    ax2.plot(df_telemetry.index, df_telemetry['total_airspeed_fps'], color='teal')
    ax2.set_ylabel("Total Airspeed, $V$ [fps]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True)

    plt.tight_layout()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Convert to degrees
    theta_deg = df_telemetry['state_attitude_angle_rad'] * (180 / np.pi)
    de_deg_s = df_telemetry['Cmd_de'] * (180 / np.pi)

    ax1.plot(df_telemetry.index, theta_deg, color='C1')
    ax1.set_ylabel("Pitch Angle, $\\theta$ [deg]")
    ax1.set_title("Pitch Dynamics over Time")
    ax1.grid(True)

    ax2.plot(df_telemetry.index, de_deg_s, color='C2')
    ax2.set_ylabel("Elevator Cmd, $/delta_e$ [deg]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True)

    plt.tight_layout()






    plt.show()