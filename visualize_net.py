import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import glob
import math
import caffe

from visualize import _Visualizer
from utils import get_net, get_inputs

class NetVisualizer(_Visualizer):

    def __init__(self, cmd, modelfile, deployfile, *args, **kwargs):
        super(NetVisualizer, self).__init__(cmd, *args, **kwargs)
        self.net = get_net(modelfile, deployfile)


class KernelVisualizer(NetVisualizer):

    def __init__(self, *args, **kwargs):
        super(KernelVisualizer, self).__init__("Kernel", *args, **kwargs)
        self.cmd = "%s for %s" %(self.cmd, args[0])  # args[0] is always the modelfile

    def data(self):
        return self.net.params["conv1"][0].data


class OutputVisualizer(NetVisualizer):

    def __init__(self, modelfile, deployfile, imagedir, image_limit, **kwargs):
        shuffle = kwargs.pop("shuffle")
        super(OutputVisualizer, self).__init__("Output", modelfile, deployfile, **kwargs)

        self.images = get_inputs(imagedir, "jpg", limit=image_limit, color=False, shuffle=shuffle)
        self.ncols = 2

    def data(self):
        scores = []
        for i, img in enumerate(self.images):
            # Add original image
            scores.append(img.reshape(img.shape[2], img.shape[0], img.shape[1]))

            # Add reconstructed image
            score = self.net.predict([img], oversample=False)[0]
            scores.append(score.copy())

        return scores


class FilterVisualizer(NetVisualizer):

    def __init__(self, imagedir, image_limit, filter_name, filter_num,
                 *args, **kwargs):
        super(FilterVisualizer, self).__init__(*args, **kwargs)

        self.images = get_inputs(imagedir, "jpg", image_limit, color=False)
        self.filter_name = filter_name
        self.filter_num = filter_num
        self.ncols = 2

    def data(self):
        scores = []
        for i, img in enumerate(self.images):
            # Add original image
            scores.append(img.reshape(img.shape[2], img.shape[0], img.shape[1]))

            # Add filter
            score = self.net.predict([img], oversample=False)[0]
            score = self.net.blobs[self.filter_name].data[0,self.filter_num, :, :]
            scores.append(score.copy())

        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile",
                        help="Model file to use")
    parser.add_argument("deployfile",
                        help="Deploy file to use")
    parser.add_argument("command",
                        help="Command to be executed")
    parser.add_argument("--images",
                        type=str,
                        help="Images to use for OutputVisualizer")
    parser.add_argument("--limit",
                        type=int,
                        default=10,
                        help="Number of output visualizations to show")
    parser.add_argument("--save",
                        type=str,
                        help="Use to save results instead of displaying them")
    parser.add_argument("--stepsize",
                        type=int,
                        default=10000,
                        help="The number of iterations to jumps when browsing models")
    parser.add_argument("--shuffle",
                        action="store_true",
                        help="Shuffles the images if using the OutputVisualizer")

    args = parser.parse_args()

    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    cmd = args.command.lower()
    if cmd == "kernel":
        viz = KernelVisualizer(args.modelfile, args.deployfile)
    elif cmd == "output":
        viz = OutputVisualizer(args.modelfile, args.deployfile, args.images,
                               args.limit, shuffle=args.shuffle)
    elif cmd == "filter":
        raise NotImplementedError("FilterVisualizer isn't yet implemented.")
        return
    else:
        raise ValueError("Invalid command.")

    viz.plot()
    if args.save is not None:
        plt.savefig(args.save, dpi=300)
    else:
        plt.show()

if __name__ == '__main__':
    main()
