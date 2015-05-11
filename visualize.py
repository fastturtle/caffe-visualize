import math
import matplotlib.pyplot as plt

class _Visualizer(object):

    def __init__(self, cmd):
        self.cmd = cmd
        self.ncols = None

    def show(self):
        self.plot()
        # raise NotImplementedError()

    def plot(self):
        fig = plt.figure(self.cmd)  # Set current figure
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
        fig.canvas.draw()

    def get_index(self, i, nrows, ncols):
        return i + 1

    def data(self):
        """ Should return a 4-d array """
        raise NotImplementedError()
