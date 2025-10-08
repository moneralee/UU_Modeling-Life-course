import numpy as np
import matplotlib.pyplot as plt

# Enable interactive mode for matplotlib
plt.ion()

# Setup figure and axis for plotting the arrow
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 600)  # x-axis limits
ax.set_ylim(0, 250)  # y-axis limits
ax.set_aspect('equal')  # Keep aspect ratio square
ax.set_facecolor('#f0f0f0')  # Background color
ax.set_title("A moving vector with an arrowhead")  # Title

# Initial position and velocity
x, y = 250.0, 180.0      # Position coordinates
vx, vy = 5.0, 10.5        # Velocity components


def draw_arrow(x, y, vx, vy):
    """
    Draws an arrow at position (x, y) with velocity (vx, vy).
    """
    ax.clear()
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 250)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    ax.set_title("A moving vector with an arrowhead")

    # Normalize velocity for drawing the arrow
    
    dx = vx*5
    dy = vy*5

    # Arrow shaft
    end_x = x + dx
    end_y = y + dy

    # Arrowhead calculation
    angle = np.arctan2(dy, dx)
    angle_offset = np.pi / 7
    hx1_x = end_x - np.cos(angle - angle_offset)
    hx1_y = end_y - np.sin(angle - angle_offset)
    hx2_x = end_x - np.cos(angle + angle_offset)
    hx2_y = end_y - np.sin(angle + angle_offset)

    # Draw shaft
    ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='#007acc', width=0.005)
    # Draw base point
    ax.plot(x, y, 'o', color='#333')

    # Labels
    ax.text(x+10, y+10, f"x = {x:.2f}")
    ax.text(x+10, y-5, f"y = {y:.2f}")
    ax.text(end_x + 10, end_y - 20, f"vₓ = {vx:.2f}")
    ax.text(end_x + 10, end_y, f"vᵧ = {vy:.2f}")

    plt.draw()
    plt.pause(0.03)

# Animation loop: update position by velocity
for i in range(500):
    x += vx*0.1  # Update x position
    y += vy*0.1  # Update y position

    # Wrap around edges
    x %= 600
    y %= 250
    
    
    draw_arrow(x, y, vx, vy)

plt.ioff()