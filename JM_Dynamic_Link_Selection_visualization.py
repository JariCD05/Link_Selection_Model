import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

class Dynamic_link_selection_visualization:
    def __init__(self, active_satellites, num_satellites, satellite_image_path = 'images/satellite.png', folder_path='animations' , filename='link_selection.mp4'):
        self.active_satellites = active_satellites
        self.num_satellites = num_satellites
        self.satellite_image_path = satellite_image_path
        self.folder_path = folder_path
        self.filename = filename

        self.num_columns = len(active_satellites)
        self.time_steps = np.arange(self.num_columns)

        self.satellite_img = Image.open(self.satellite_image_path)
        self.satellite_img = self.satellite_img.resize((200, 200))

        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.ax.set_xlim(0, self.num_columns)
        self.ax.set_ylim(-1, self.num_satellites)
        self.line, = self.ax.plot([], [], lw=2)
        self.satellite_icon = self.ax.imshow(self.satellite_img, extent=(0, 1, -1, 0))

    def init_animation(self):
        self.line.set_data([], [])
        self.satellite_icon.set_extent((0, 0, -1, -1))
        return self.line, self.satellite_icon,

    def update_animation(self, frame):
        xdata = self.time_steps[:frame]
        ydata = [-1 if i >= len(self.active_satellites) or self.active_satellites[i] == "No link" else int(self.active_satellites[i]) for i in range(frame)]
        self.line.set_data(xdata, ydata)

        if ydata:
            satellite_position = ydata[-1]
            if frame < self.num_columns:
                self.satellite_icon.set_extent((frame - 0.5, frame + 0.5, satellite_position - 0.5, satellite_position + 0.5))
        else:
            self.satellite_icon.set_extent((0, 0, 0, 0))
        return self.line, self.satellite_icon

    def run(self):
        ani = FuncAnimation(self.fig, self.update_animation, frames=self.num_columns, init_func=self.init_animation, blit=True, repeat=False, interval=400)
        plt.title('Link selection')
        plt.xlabel('Time Step')
        plt.ylabel('Link selected')
        plt.yticks(range(-1, self.num_satellites), ['No link'] + [f'Sat {i+1}' for i in range(self.num_satellites)])
        plt.grid(True)

        full_path = f"{self.folder_path}/{self.filename}"
        ani.save(full_path, writer='ffmpeg', fps=2.5)
        #plt.show()

# Example usage:
# visualization = DynamicLinkSelectionVisualisation(active_satellites=array_of_active_satellites, num_satellites=10, satellite_image_path="satellite.png")
# visualization.run()
