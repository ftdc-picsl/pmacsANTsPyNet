#!/usr/bin/env python

import argparse
import sys

import os.path
from os import path

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 prog='shiva_pvs_segmentation', add_help = False, description='''
Wrapper for shiva_pvs_segmentation in AntsPyNet.


If you use this script, please cite the following paper for ANTsPyNet:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological and medical imaging.
Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6

and the SHIVA paper

    https://pubmed.ncbi.nlm.nih.gov/34262443/

original implementation available here:

    https://github.com/pboutinaud/SHIVA_PVS


''')
required = parser.add_argument_group('Required arguments')
required.add_argument('--flair-image', help='Input FLAIR image to segment', type=str, required=True)
required.add_argument('-o', '--output', help='Output file', type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument('-h', '--help', action='help', help='show this help message and exit')
optional.add_argument('-t', '--threads', help='Number of threads in tensorflow operations. Use environment variable ' \
                    'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS to control threading in ANTs calls', type=int, default=1)
required.add_argument('--t1w-image', help='Optional T1w image to segment, must be aligned FLAIR.', type=str, required=False)

args = parser.parse_args()


# Now fire up ants etc
import ants
import antspynet
import tensorflow as tf

# Internal variables for args
t1w_fn = args.t1w_image
flair_fn = args.flair_image
output_file = args.output
threads = args.threads

tf.keras.backend.clear_session()
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

flair = ants.image_read(flair_fn)

t1w = None

if t1w_fn is not None:
    t1w = ants.image_read(t1w_fn)

wmh = antspynet.shiva_pvs_segmentation(flair, t1w, which_model = "all", verbose=True)

ants.image_write(wmh, output_file)

