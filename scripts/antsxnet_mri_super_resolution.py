#!/usr/bin/env python

import argparse
import sys

import os.path
from os import path

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 prog="brain_extraction", add_help = False, description='''
Wrapper for mri_super_resolution in AntsPyNet. This code uses a lot of memory, so you might need to
brain-extract the image and then crop the FOV. You can do this with ANTs ExtractRegionFromImageByMask.

For more details of the algorithm and available options, see

https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/mri_super_resolution.py

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological and medical imaging.
Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6
''')
required = parser.add_argument_group('Required arguments')
required.add_argument("-a", "--anatomical-image", help="Input image to upsample", type=str, required=True)
required.add_argument("-o", "--output", help="Output file", type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
optional.add_argument("-t", "--threads", help="Number of threads in tensorflow operations. Use environment variable " \
                    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS to control threading in ANTs calls", type=int, default=1
                    )
args = parser.parse_args()


# Now fire up ants etc
import ants
import antspynet
import tensorflow as tf

# Internal variables for args
anatomical_image = args.anatomical_image
output_file = args.output
threads = args.threads

tf.keras.backend.clear_session()
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

anat = ants.image_read(anatomical_image)

output = antspynet.mri_super_resolution(anat, verbose=True)

ants.image_write(output, output_file)

