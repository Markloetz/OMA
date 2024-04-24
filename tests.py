import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make space for the slider

# Plot the initial data
line, = ax.plot(x, y)

# Add a slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Freq', 0.1, 10.0, valinit=1)

# Variable to store slider value
slider_value = slider.val

# Update function for the slider
def update(val):
    global slider_value  # Declare slider_value as global so we can modify it
    slider_value = val
    line.set_ydata(np.sin(slider_value * x))
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()

# Now you can use slider_value elsewhere in your code
print("Slider value:", slider_value)