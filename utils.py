import numpy as np
import os.path
import caffe
import glob

def get_inputs(input_file, ext, limit=None, **kwargs):
    input_file = os.path.expanduser(input_file)
    if input_file.endswith('npy'):
        inputs = np.load(input_file)
    elif os.path.isdir(input_file):
        fnames = sorted(glob.glob(input_file + '/*.' + ext), key=lambda x: int(x.rsplit("/")[-1].split(".")[0]))
        if limit is not None:
            fnames = fnames[:limit]
        inputs =[caffe.io.load_image(im_f, **kwargs)
                 for im_f in fnames]
    else:
        inputs = [caffe.io.load_image(input_file, **kwargs)]
    return inputs

def get_net(modelfile, deployfile, **kwargs):
    if not os.path.isfile(deployfile):
        raise IOError("File not found: %s" % deployfile)
    if not os.path.isfile(modelfile):
        raise IOError("File not found %s" % modelfile)
    return caffe.Classifier(deployfile, modelfile, **kwargs)
