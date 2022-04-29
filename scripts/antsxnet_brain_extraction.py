#!/usr/bin/env python

import argparse
import sys

import os.path
from os import path

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 prog="brain_extraction", add_help = False, description='''
Wrapper for brain extraction script in AntsPyNet.

For more details of the algorithm and available options, see

https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/brain_extraction.py

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological and medical imaging.
Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6
''')
required = parser.add_argument_group('Required arguments')
required.add_argument("-a", "--head-image", help="Input image to be brain extracted", type=str, required=True)
required.add_argument("-m", "--modality", help='''Image modality. Options are
    * "t1": T1-weighted MRI---ANTs-trained.  Previous versions are specified as "t1.v0", "t1.v1".
    * "t1nobrainer": T1-weighted MRI---FreeSurfer-trained: h/t Satra Ghosh and Jakub Kaczmarzyk.
    * "t1combined": Brian's combination of "t1" and "t1nobrainer".  One can also specify
      "t1combined[X]" where X is the morphological radius.  X = 12 by default.
    * "flair": FLAIR MRI.
    * "t2": T2 MRI.
    * "bold": 3-D BOLD MRI.
    * "fa": Fractional anisotropy.
    * "t1t2infant": Combined T1-w/T2-w infant MRI h/t Martin Styner.
    * "t1infant": T1-w infant MRI h/t Martin Styner.
    * "t2infant": T2-w infant MRI h/t Martin Styner.
 ''', type=str, required=True)
required.add_argument("-o", "--output", help="Mask output file", type=str, required=True)
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
head_file = args.head_image
modality = args.modality
output_file = args.output
threads = args.threads

tf.keras.backend.clear_session()
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

anat = ants.image_read(head_file)

# Can be binary or probability depending on method
be_output = antspynet.brain_extraction(anat, modality=modality, verbose=True)

brain_mask = ants.iMath_get_largest_component(
          ants.threshold_image(be_output, 0.5, 1.5))

ants.image_write(brain_mask, output_file)

