import numpy as np
import matplotlib.pyplot as plt

sim_time = 10.0
dt = 0.1 # Step Time [s]


x0      = 0.0 # Position        [ft]
vx0     = 0.0 # Velocity        [ft/s]
ax0     = 1.0 # Acceleration    [ft/s^2]

time  = np.arange(0, sim_time, dt)

x_kin   = np.zeros_like(time)
vx_kin  = np.zeros_like(time)
ax_kin  = np.zeros_like(time)

x_int   = np.zeros_like(time)
vx_int  = np.zeros_like(time)
ax_int  = np.zeros_like(time)



for i, t in enumerate(time):
    x_kin[i]    = x0 + vx0 * t + 0.5 * ax0 * t ** 2
    vx_kin[i]   = vx0 + ax0 * t 
    ax_kin[i]   = ax0

    if i == 0:
        x_int[i]    = x0
        vx_int[i]   = vx0
        ax_int[i]   = ax0

    else:
        x_int[i]    = x_int[i-1] + vx_int[i-1] * dt
        vx_int[i]   = vx_int[i-1] + ax_int[i-1] * dt
        ax_int[i]   = ax_int[i-1]

plt.figure('Constant Acceleration Trajectory')

plt.subplot(3,1,1)
plt.title('Constant Acceleration Trajectory')
plt.plot(time, x_kin)
plt.plot(time, x_int)
plt.ylabel('Position [ft]')
plt.legend('Kin', 'Int')
plt.grid()

plt.subplot(3,1,2)
plt.plot(time, vx_kin)
plt.plot(time, vx_int)
plt.ylabel('Velocity [ft/s]')
plt.legend('Kin', 'Int')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, ax_kin)
plt.plot(time, ax_int)
plt.ylabel('Acceleration [ft/s^2]')
plt.xlabel('Time [s]')
plt.grid()

plt.show()
