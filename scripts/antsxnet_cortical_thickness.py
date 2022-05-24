#!/usr/bin/env python

import argparse
import os
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 prog="cortical_thickness", add_help=False, description='''
Wrapper for cortical thickness in ANTsPyNet.

For more details of the algorithm and available options, see

https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/cortical_thickness.py

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological
and medical imaging. Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6
''')
required = parser.add_argument_group('Required arguments')
required.add_argument("-a", "--head-image", help="Input T1w image", type=str, required=True)
required.add_argument("-o", "--output-root", help="Full path to output including file prefix", type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("-d", "--dkt-labeling", help='''Optionally include deep DKT31 parcelation''',
                                                 type=bool, default=True)
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

deep_ct = antspynet.cortical_thickness(anat, verbose=True)

output_file_suffixes = {'thickness_image' : 'CorticalThickness.nii.gz',
                        'segmentation_image' : 'BrainSegmentation.nii.gz',
                        'csf_probability_image' : 'BrainSegmentationPosteriors1.nii.gz',
                        'gray_matter_probability_image' : 'BrainSegmentationPosteriors2.nii.gz',
                        'white_matter_probability_image' : 'BrainSegmentationPosteriors3.nii.gz',
                        'deep_gray_matter_probability_image' : 'BrainSegmentationPosteriors4.nii.gz',
                        'brain_stem_probability_image' : 'BrainSegmentationPosteriors5.nii.gz',
                        'cerebellum_probability_image' : 'BrainSegmentationPosteriors6.nii.gz'
                        }

for key, image in deep_ct.items():
    output_file = args.output_root + output_file_suffixes.get(key)
    print(f'Writing {key} as {output_file}')
    ants.image_write(image, output_file)

if args.dkt_labeling:
    print("Deep DKT parcellation\n")
    dkt_file = args.output_root + "DKT.nii.gz"
    dkt = None
    dkt = antspynet.desikan_killiany_tourville_labeling(anat, do_preprocessing=True, verbose=True)
    ants.image_write(dkt, dkt_file)

    print("DKT Propagation\n")

    dkt_prop_file = args.output_root + "DKTPropagatedLabels.nii.gz"
    dkt_mask = ants.threshold_image(dkt, 1000, 3000, 1, 0)
    dkt = dkt_mask * dkt
    ants_tmp = ants.threshold_image(deep_ct.get('thickness_image'), 0, 0, 0, 1)
    ants_dkt = ants.iMath(ants_tmp, "PropagateLabelsThroughMask", ants_tmp * dkt)
    ants.image_write(ants_dkt, dkt_prop_file)

