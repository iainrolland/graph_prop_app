import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from sklearn.neighbors import kneighbors_graph
import rasterio as rio
import matplotlib.pyplot as plt

from adj_utils import udlr
import diffusion


class Paint(object):
    DEFAULT_PEN_SIZE = 40.0
    DEFAULT_COLOR = 'black'
    SCALE = 1.5

    def __init__(self):
        self.root = tk.Tk()
        self.low = None
        self.high = None

        self.partial_og = np.rollaxis(rio.open("data/bx_2.tif").read(), 0, 3).astype(np.float32)
        self.reference_og = np.rollaxis(rio.open("data/bx_0.tif").read(), 0, 3).astype(np.float32)

        height, width = self.partial_og.shape[:2]

        self.pen_button = tk.Button(self.root, text='draw', command=self.use_pen)
        self.pen_button.grid(row=0, column=4, columnspan=2)

        self.eraser_button = tk.Button(self.root, text='erase', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=6)

        self.choose_size_button = tk.Scale(self.root, from_=3, to=200, orient=tk.HORIZONTAL)
        self.choose_size_button.grid(row=0, column=7)

        self.pil_mask = Image.new("RGB", (width, height), (255, 255, 255))
        self.pil_draw = ImageDraw.Draw(self.pil_mask)

        self.reference_canvas = tk.Canvas(self.root, bg='white', width=width * self.SCALE, height=height * self.SCALE)
        self.reference_canvas.grid(row=1, columnspan=4)
        self.draw_background("reference", self.reference_og, self.reference_canvas)

        self.partial_canvas = tk.Canvas(self.root, bg='white', width=width * self.SCALE, height=height * self.SCALE)
        self.partial_canvas.grid(row=1, columnspan=4, column=4)
        self.draw_background("partial", self.partial_og, self.partial_canvas)

        self.diffused_canvas = tk.Canvas(self.root, bg='white', width=width * self.SCALE, height=height * self.SCALE)
        self.diffused_canvas.grid(row=1, columnspan=4, column=8)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.adjacency = None
        self.old_x = None
        self.old_y = None
        self.choose_size_button.set(self.DEFAULT_PEN_SIZE)
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.partial_canvas.bind('<B1-Motion>', self.paint)
        self.partial_canvas.bind('<ButtonRelease-1>', self.reset)

    def draw_background(self, name, array, canvas):
        if isinstance(array, np.ndarray):
            rgb = array.copy()
            if rgb.shape[-1] > 3:
                rgb = rgb[..., [2, 1, 0]]
            if rgb.dtype is not np.uint8:
                if self.low is None or self.high is None:
                    self.low, self.high = np.percentile(rgb.reshape(-1, rgb.shape[-1]), q=[1, 99])
                    self.low -= .2 * (self.high - self.low)
                    self.high += .2 * (self.high - self.low)
                rgb = np.clip((rgb - self.low) / (self.high - self.low), 0, 1)
                rgb = (255 * rgb).astype(np.uint8)
            self.__setattr__(name, ImageTk.PhotoImage(
                image=Image.fromarray(rgb).resize([int(self.partial_og.shape[1] * self.SCALE),
                                                   int(self.partial_og.shape[0] * self.SCALE)], Image.ANTIALIAS)))
        if hasattr(self, name):
            canvas.create_image(0, 0, image=self.__getattribute__(name), anchor="nw")

    def masked_partial(self):
        array = self.partial_og.copy()  # take the original but make it black where we have drawn it so
        array[np.asarray(self.pil_mask)[..., 0] == 0] = 0
        return array

    def graph_prop_completion(self):
        if self.adjacency is None:
            reference_tensor = self.reference_og.copy()
            self.adjacency = kneighbors_graph(reference_tensor.reshape(-1, reference_tensor.shape[2]), 10,
                                              include_self=False)
            self.adjacency = self.adjacency + self.adjacency.T  # to make graph symmetric (using k neighbours in "either" rather than "mutual" mode)
            self.adjacency[self.adjacency > 1] = 1  # get rid of any edges we just made double
            self.adjacency += udlr(reference_tensor.shape[:2]) * 3

        array = self.partial_og.copy().astype(float)  # make into float (in case integer diffusion weird)
        if np.sum(np.asarray(self.pil_mask)[..., 0] == 0) != 0:  # if >= 1 pixel masked
            array[np.asarray(self.pil_mask)[..., 0] == 0] = 0
            array = diffusion.graph_prop(self.adjacency, array, (np.asarray(self.pil_mask)[..., 0] == 255).astype(int),
                                     iterative=False)
        return array

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=tk.RAISED)
        some_button.config(relief=tk.SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.partial_canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                            width=self.line_width, fill=paint_color, capstyle=tk.ROUND, smooth=tk.TRUE,
                                            splinesteps=36)
            self.pil_draw.line(
                [(self.old_x / self.SCALE, self.old_y / self.SCALE), (event.x / self.SCALE, event.y / self.SCALE)],
                fill=paint_color,
                width=int(self.line_width / self.SCALE),
                joint="curve")
            self.pil_draw.ellipse(
                [(self.old_x - self.line_width / 2) / self.SCALE, (self.old_y - self.line_width / 2) / self.SCALE,
                 (self.old_x + self.line_width / 2) / self.SCALE, (self.old_y + self.line_width / 2) / self.SCALE],
                outline=None,
                fill=paint_color)
        self.old_x = event.x
        self.old_y = event.y
        if self.eraser_on:
            self.partial_canvas.delete("all")
            self.draw_background(
                "original",
                self.masked_partial(),
                self.partial_canvas
            )

    def reset(self, event):
        self.old_x, self.old_y = None, None
        # self.diffused_canvas.delete("all")
        self.label = tk.Label(self.root, text="Propagating...", font="Helvetica 18 bold")
        self.label.grid(row=0, column=8, columnspan=4)
        # if hasattr(self, "graph_prop"):
        #     self.draw_background(
        #         "diffused",
        #         None,
        #         self.diffused_canvas
        #     )
        self.root.after(10, self.update)  # after 10ms compute the diffused completion

    def update(self):
        self.draw_background(
            "diffused",
            self.graph_prop_completion(),
            self.diffused_canvas
        )
        self.label.config(text="Complete!")


if __name__ == '__main__':
    Paint()
