import numpy as np
import sys
import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import caffe

from utils import get_net


# def _get_net(net_path):
#     net = caffe.Classifier("deploy.prototxt", net_path)
#     return net

def all_nets(snapshot_dir, deployfile):
    for fname in sorted(os.listdir(snapshot_dir),
        key=lambda x: int(x.rsplit(".", 1)[0].rsplit("_", 1)[1])):
        if fname.endswith(".caffemodel"):
            name, ext = fname.rsplit(".", 1)
            prefix, niters = name.rsplit("_", 1)
            niters = int(niters)
            yield niters, get_net(os.path.join(snapshot_dir, fname),
                                  deployfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_dir",
                        help="Snapshots directory to use")

    parser.add_argument("deployfile",
                        help="Deploy file to use")

    parser.add_argument("layer",
                        help="Layer name to use")

    parser.add_argument("--save",
                        type=str,
                        help="Use deep prefix for neural network")

    args = parser.parse_args()
    # return

    iters = []
    params = defaultdict(list)
    for niters, net in all_nets(args.snapshot_dir, args.deployfile):
        iters.append(niters)

        iter_params = net.params[args.layer][0].data
        for filter_num in xrange(len(iter_params)):
            params[filter_num].append(iter_params[filter_num].flatten())

        # iter_params = net.params["deconv1"][0].data
        # for filter_num in xrange(len(iter_params)):
        #   params[filter_num + len(iter_params)].append(iter_params[filter_num].flatten())

    for filter_num, filter_params in params.iteritems():
        ax = plt.subplot(4, 4, filter_num)
        ax.set_title("Filter %d" % filter_num)
        plt.plot(iters, filter_params)

    if args.save is not None:
        plt.savefig(args.save, dpi=300)
    else:
        plt.show()

if __name__ == '__main__':
    main()
