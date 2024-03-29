#!/usr/bin/env python

import argparse
import os
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 prog="desikan_killiany_tourville_labeling", add_help=False, description='''
Wrapper for DKT labeling in ANTsPyNet.

For more details of the algorithm and available options, see

https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/desikan_killiany_tourville_labeling.py

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological
and medical imaging. Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6
''')
required = parser.add_argument_group('Required arguments')
required.add_argument("-a", "--head-image", help="Input T1w image", type=str, required=True)
required.add_argument("-o", "--output-root", help="Full path to output including file prefix", type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
optional.add_argument("-t", "--threads", help="Number of threads in tensorflow operations. Use environment variable "
                    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS to control threading in ANTs calls", type=int, default=1
                    )
args = parser.parse_args()


import ants
import antspynet
import tensorflow as tf

output_dir = os.path.dirname(args.output_root)
output_file_prefix = os.path.basename(args.output_root)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tf.keras.backend.clear_session()
tf.config.threading.set_intra_op_parallelism_threads(args.threads)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)

anat = ants.image_read(args.head_image)

deep_dkt = antspynet.desikan_killiany_tourville_labeling(anat, verbose=True)
ants.image_write(deep_dkt, args.output_root + "DKT.nii.gz")
