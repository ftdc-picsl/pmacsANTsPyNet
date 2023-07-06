#!/usr/bin/env python

import argparse
import numpy as np
import os
import sys
import textwrap

class RawDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass

parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter,
                    prog="brain_segmentation", add_help=False,
                    description = f'''
Wrapper for cerebellum_morphology in ANTsPyNet.

For more details of the algorithm and available options, see

https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/cerebellum_morphology.py

Preprocessing
-------------

Input images should be preprocessed. By default, this is performed by the built-in preprocessing
function. Advanced users can turn this off. The preprocessing for this pipeline is N4 bias correction.

Citation
--------

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological
and medical imaging. Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6

'''
                                )
required = parser.add_argument_group('Required arguments')
required.add_argument("-a", "--anatomical-image", help="Input T1w image", type=str, required=True)
required.add_argument("-o", "--output-root", help="Full path to output including file prefix", type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
optional.add_argument("-k", "--compute-thickness", action='store_true', help="Compute thickness of cerebellar gray matter")
optional.add_argument("-m", "--cerebellum-mask", help="Optional cerebellum mask. If not provided, will be estimated automatically",
                      type=str)
optional.add_argument("--no-preprocess", action='store_true', help="Skip internal preprocessing (N4).")
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

anat = ants.image_read(args.anatomical_image)

# The results to write to disk
output_seg = None

do_preproc = not args.no_preprocess

output_seg = antspynet.cerebellum_morphology(anat, do_preprocessing=do_preproc, compute_thickness_image=args.compute_thickness, verbose=True)

# Write output to disk
seg_output_file = f'{args.output_root}CerebellumSegmentation.nii.gz'

ants.image_write(output_seg['parcellation_segmentation_image'], f'{args.output_root}CerebellumParcellation.nii.gz')

for idx, image in enumerate(output_seg['parcellation_probability_images']):
    ants.image_write(image, f'{args.output_root}CerebellumParcellationPosteriors{idx}.nii.gz')


ants.image_write(output_seg['tissue_segmentation_image'], f'{args.output_root}CerebellumSegmentation.nii.gz')

for idx, image in enumerate(output_seg['tissue_probability_images']):
    ants.image_write(image, f'{args.output_root}CerebellumSegmentationPosteriors{idx}.nii.gz')

if args.compute_thickness:
    ants.image_write(output_seg['thickness_image'], f'{args.output_root}CerebellumThickness.nii.gz')


