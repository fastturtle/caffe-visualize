import math
import matplotlib.pyplot as plt

class _Visualizer(object):

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.ncols = None
        self.kwargs = kwargs

    def show(self):
        self.plot()
        # raise NotImplementedError()

    def plot(self):
        fig = plt.figure(self.cmd, figsize=(10, 10))  # Set current figure
        plt.suptitle(self.cmd.title())

        data = self.data()
        n = len(data)
        if self.ncols == None:
            # Fit the data to a square grid
            ncols = int(math.sqrt(n))
            nrows = n / ncols + n % ncols
        else:
            ncols = self.ncols
            nrows = n / ncols

        for i, d in enumerate(data):
            idx =  self.get_index(i, nrows, ncols)
            ax = fig.add_subplot(nrows, ncols, idx)
            axes_images = ax.get_images()
            if len(d.shape) > 2:
                d = d.reshape(d.shape[1], d.shape[2])  # Remove color channel
            if len(axes_images) > 0:
                axes_images[0].set_data(d)
            else:
                plt.imshow(d)
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(ax.get_xticklabels(), visible=False)
                title = self.subplot_title(idx, nrows, ncols)
                if title is not None:
                    ax.set_title(title, fontsize=10)

        if "tight_layout" in self.kwargs and self.kwargs["tight_layout"]:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig.canvas.draw()

    def get_index(self, i, nrows, ncols):
        return i + 1

    def subplot_title(self, i, nrows, ncols):
        return None

    def data(self):
        """ Should return a 4-d array """
        raise NotImplementedError()
