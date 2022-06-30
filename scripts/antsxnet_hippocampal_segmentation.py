#!/usr/bin/env python

import argparse
import sys

import os.path
from os import path

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 prog="brain_extraction", add_help = False, description='''
Wrapper for deep_flash in AntsPyNet.

    Hippocampal/Enthorhinal segmentation using "Deep Flash"
    Perform hippocampal/entorhinal segmentation in T1 and T1/T2 images using
    labels from Mike Yassa's lab
    https://faculty.sites.uci.edu/myassa/

    The labeling is as follows:
    Label 0 :  background
    Label 5 :  left aLEC
    Label 6 :  right aLEC
    Label 7 :  left pMEC
    Label 8 :  right pMEC
    Label 9 :  left perirhinal
    Label 10:  right perirhinal
    Label 11:  left parahippocampal
    Label 12:  right parahippocampal
    Label 13:  left DG/CA2/CA3/CA4
    Label 14:  right DG/CA2/CA3/CA4
    Label 15:  left CA1
    Label 16:  right CA1
    Label 17:  left subiculum
    Label 18:  right subiculum

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological and medical imaging.
Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6
''')
required = parser.add_argument_group('Required arguments')
required.add_argument("--t1w", help="T1w input image", type=str, required=True)
required.add_argument("--t2w", help="T2w input image, aligned to T1w", type=str, required=False)
required.add_argument("-o", "--output", help="Output file", type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
optional.add_argument("-t", "--threads", help="Number of threads in tensorflow operations. Use environment variable " \
                    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS to control threading in ANTs calls", type=int, default=1
                    )
optional.add_argument("--use-rank-intensity", help="Use a rank intensity transform on the cropped ROI.", action='store_true', required=False, default=True)
optional.add_argument('--no-use-rank-intensity', help="Use histogram matching on the cropped template ROI.", dest='use-rank-intensity', action='store_false')
args = parser.parse_args()


import ants
import antspynet
import tensorflow as tf

# Internal variables for args
t1w = args.t1w
t2w = args.t2w
output_file = args.output
threads = args.threads

tf.keras.backend.clear_session()
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

t1w_image = ants.image_read(t1w)
t2w_image = None

if (t2w is not None):
    t2w_image = ants.image_read(t2w)

output = antspynet.deep_flash(t1w_image, t2w_image, use_rank_intensity=args.use_rank_intensity, verbose=True)

ants.image_write(output['segmentation_image'], output_file)

