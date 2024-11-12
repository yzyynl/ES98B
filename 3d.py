import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
M_E = 5.972e24  # Earth's mass, kg
R_E = 6371e3  # Earth's radius, m
rho_0 = 1.225  # Atmospheric density at sea level, kg/m^3
H = 8500  # Scale height for atmospheric density, m
C_d = 2.2  # Drag coefficient
A = 10  # Satellite cross-sectional area, m^2
mass = 500  # Satellite mass, kg
initial_altitude = 150e3  # Initial altitude, 150 km
initial_velocity = np.sqrt(G * M_E / (R_E + initial_altitude))  # Initial velocity set to orbital speed
time_step = 1  # Time step, 1 second
total_time = 300000  # Total simulation time, 300000 seconds

# Initial state
r = R_E + initial_altitude
x, y = 0, r  # 2D case, starting position along the y-axis
vx, vy = initial_velocity, 0  # Initial velocity along the x direction

# Data storage
trajectory = []

# Function to calculate atmospheric drag and gravitational acceleration
def calculate_acceleration(x, y, vx, vy):
    r = np.sqrt(x ** 2 + y ** 2)
    h = r - R_E  # Current altitude
    rho = rho_0 * np.exp(-h / H) if h > 0 else 0  # Atmospheric density decays with height

    # Gravitational acceleration
    gravity = -G * M_E / r ** 3
    ax_g, ay_g = gravity * x, gravity * y

    # Speed and atmospheric drag
    v = np.sqrt(vx ** 2 + vy ** 2)
    drag = 0.5 * rho * v * C_d * A / mass
    ax_d, ay_d = -drag * vx, -drag * vy

    # Total acceleration
    ax = ax_g + ax_d
    ay = ay_g + ay_d

    return ax, ay

# Fourth-order Runge-Kutta (RK4) integration
def rk4_step(x, y, vx, vy, dt):
    ax1, ay1 = calculate_acceleration(x, y, vx, vy)
    dx1, dy1 = vx * dt, vy * dt
    dvx1, dvy1 = ax1 * dt, ay1 * dt

    ax2, ay2 = calculate_acceleration(x + dx1 / 2, y + dy1 / 2, vx + dvx1 / 2, vy + dvy1 / 2)
    dx2, dy2 = (vx + dvx1 / 2) * dt, (vy + dvy1 / 2) * dt
    dvx2, dvy2 = ax2 * dt, ay2 * dt

    ax3, ay3 = calculate_acceleration(x + dx2 / 2, y + dy2 / 2, vx + dvx2 / 2, vy + dvy2 / 2)
    dx3, dy3 = (vx + dvx2 / 2) * dt, (vy + dvy2 / 2) * dt
    dvx3, dvy3 = ax3 * dt, ay3 * dt

    ax4, ay4 = calculate_acceleration(x + dx3, y + dy3, vx + dvx3, vy + dvy3)
    dx4, dy4 = (vx + dvx3) * dt, (vy + dvy3) * dt
    dvx4, dvy4 = ax4 * dt, ay4 * dt

    x_next = x + (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6
    y_next = y + (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6
    vx_next = vx + (dvx1 + 2 * dvx2 + 2 * dvx3 + dvx4) / 6
    vy_next = vy + (dvy1 + 2 * dvy2 + 2 * dvy3 + dvy4) / 6

    return x_next, y_next, vx_next, vy_next

# Main simulation loop
for step in range(int(total_time / time_step)):
    trajectory.append((x, y))  # Record trajectory coordinates

    # Update position and velocity
    x, y, vx, vy = rk4_step(x, y, vx, vy, time_step)

    # Check if satellite reaches the ground
    if np.sqrt(x ** 2 + y ** 2) <= R_E:
        print(f"Simulation ended: Satellite has re-entered the atmosphere at step {step}.")
        break

# Extract trajectory data for visualization
x_vals = [pos[0] for pos in trajectory]
y_vals = [pos[1] for pos in trajectory]

# Visualize trajectory plot
plt.figure(figsize=(8, 8))
plt.plot(x_vals, y_vals, label="Satellite Trajectory", color="purple")
circle = plt.Circle((0, 0), R_E, color="green", alpha=0.3, label="Earth")
plt.gca().add_artist(circle)
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Satellite XY Plane Projection")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()

