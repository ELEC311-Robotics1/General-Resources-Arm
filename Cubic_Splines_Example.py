from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

t_wp = [0, 1, 2, 3, 4]  # time waypoints
x_wp = [0, 1, 0, -1, 0]
y_wp = [0, 0, 1, 1, 0]

cx = CubicSpline(t_wp, x_wp, bc_type='natural')
cy = CubicSpline(t_wp, y_wp, bc_type='natural')

t = np.linspace(0, 4, 100)

x   = cx(t)        # position
dot_x  = cx(t, 1)     # velocity
ddot_x = cx(t, 2)     # acceleration

# Plot x vs t and x_wp vs t
plt.figure()
plt.plot(t, x, label='Cubic Spline')
plt.plot(t_wp, x_wp, 'ro', label='Waypoints')
plt.title('Cubic Spline Interpolation of x(t)')
plt.legend()
plt.show()
