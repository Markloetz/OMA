from OMA import OMA_Module as oma
import numpy as np
import scipy
from matplotlib import pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk, ImageSequence


class GifPlayer(tk.Label):
    def __init__(self, master, filename):
        im = Image.open(filename)
        self.sequence = [ImageTk.PhotoImage(img)
                         for img in ImageSequence.Iterator(im)]
        try:
            self.delay = im.info['duration']
        except KeyError:
            self.delay = 100
        self.idx = 0

        tk.Label.__init__(self, master, image=self.sequence[0])
        self.image = self.sequence[0]
        self.after(self.delay, self.play)

    def play(self):
        self.idx = (self.idx + 1) % len(self.sequence)
        self.config(image=self.sequence[self.idx])
        self.image = self.sequence[self.idx]
        self.after(self.delay, self.play)

if __name__ == "__main__":
    # Testing the loading and playback of gifs
    root = tk.Tk()
    player = GifPlayer(root, 'Animations/Plate/mode_0_20Hz.gif')
    player.pack()
    root.mainloop()