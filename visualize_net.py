import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import glob
import math
import caffe

from visualize import _Visualizer
from utils import get_net

class NetVisualizer(_Visualizer):

    def __init__(self, net, *args, **kwargs):
        super(NetVisualizer, self).__init__(*args, **kwargs)
        self.net = net


class KernelVisualizer(NetVisualizer):

    def data(self):
        return self.net.params["conv1"][0].data


class OutputVisualizer(NetVisualizer):

    def __init__(self, imagedir, image_limit, *args, **kwargs):
        super(OutputVisualizer, self).__init__(*args, **kwargs)

        self.images = get_inputs(imagedir, "jpg", image_limit, color=False)
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
    # parser.add_argument("nstates",
    #                     type=int,
    #                     help="Number of states. Used to determine image folder.")
    # parser.add_argument("niters",
    #                     type=int,
    #                     help="Number of iterations for snapshot")
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

    args = parser.parse_args()

    # Remove trailing "/" from path
    # basedir = args.base_dir[:-1] if args.base_dir[-1] == "/" else args.base_dir

    # imgs = get_images("%s/images/%dstates" % (basedir, args.nstates),
    #                   args.output_limit)

    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    
    viz = KernelVisualizer(get_net(args.modelfile, args.deployfile), cmd="Test")
    viz.plot()

    if args.save is not None:
        plt.savefig(args.save)
    else:
        plt.show()

if __name__ == '__main__':
    main()