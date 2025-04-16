import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load CSV file.
# Assumes columns: 0=time, 1=x, 2=y, 3=z; adjust as needed.
data = pd.read_csv('positions_log.csv', header=None)
x = data[1].values  # x positions
y = data[2].values  # y positions
z = data[3].values  # z positions

# Create a 3D plot.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis labels and title.
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory Recreation')

# Plot the full trajectory as a reference (static line).
ax.plot(x, y, z, label='Trajectory', color='blue', linewidth=1, alpha=0.3)

# Highlight the start and end points.
ax.scatter(x[0], y[0], z[0], color='green', label='Start', s=100)  # Start point
ax.scatter(x[-1], y[-1], z[-1], color='red', label='End', s=100)   # End point

# Create a moving point to represent the end-effector.
moving_point, = ax.plot([], [], [], 'ro', label='End-Effector', markersize=8)

# Initialize the animation.
def init():
    moving_point.set_data([], [])
    moving_point.set_3d_properties([])
    return moving_point,

# Update function for the animation.
def update(frame):
    # Wrap x[frame], y[frame], and z[frame] in lists to make them sequences.
    moving_point.set_data([x[frame]], [y[frame]])
    moving_point.set_3d_properties([z[frame]])
    return moving_point,

# Create the animation with a 0.5-second interval.
ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=500)

# Add legend and show the plot.
ax.legend()
plt.show()