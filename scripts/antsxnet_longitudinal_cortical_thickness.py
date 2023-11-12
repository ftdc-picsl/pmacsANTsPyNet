#!/usr/bin/env python

import argparse
import os
import re
import shutil
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 prog="longitudinal_cortical_thickness", add_help=False, description='''
Wrapper for longitudinal cortical thickness in ANTsPyNet.

For more details of the algorithm and available options, see

https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/cortical_thickness.py

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological
and medical imaging. Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6
''')
required = parser.add_argument_group('Required arguments')
required.add_argument("-a", "--head-images", help="Input T1w images", nargs='*', type=str, required=True)
required.add_argument("-o", "--output-dir", help="Full path to output directory", type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("--initial-sst", help="Single subject template image or string to initialize from "
                      "ANTsPyNet built-in options (eg, 'oasis')", type=str, default='oasis')
optional.add_argument("--sst-iterations", help="SST iterations, set to 0 to use initial SST", type=int, default=1)
optional.add_argument("--sst-transform", help="SST transform. Used to build SST, but final transform to SST space is affine, set to 0 to use initial SST", type=int, default=1)
optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
optional.add_argument("-t", "--threads", help="Number of threads in tensorflow operations. Use environment variable "
                    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS to control threading in ANTs calls", type=int, default=1
                    )
args = parser.parse_args()

import ants
import antspynet
import tensorflow as tf

output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tf.keras.backend.clear_session()
tf.config.threading.set_intra_op_parallelism_threads(args.threads)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)

anat_images = [ ants.image_read(i) for i in args.head_images ]

deep_ct = antspynet.longitudinal_cortical_thickness(anat_images, initial_template=args.initial_sst,
                                                    number_of_iterations=args.sst_iterations,
                                                    refinement_transform=args.sst_transform, verbose=True)

# Output suffixes for image data
# Handle warps separately
output_file_suffixes = {'thickness_image' : 'CorticalThickness.nii.gz',
                        'segmentation_image' : 'BrainSegmentation.nii.gz',
                        'csf_probability_image' : 'BrainSegmentationPosteriors1.nii.gz',
                        'gray_matter_probability_image' : 'BrainSegmentationPosteriors2.nii.gz',
                        'white_matter_probability_image' : 'BrainSegmentationPosteriors3.nii.gz',
                        'deep_gray_matter_probability_image' : 'BrainSegmentationPosteriors4.nii.gz',
                        'brain_stem_probability_image' : 'BrainSegmentationPosteriors5.nii.gz',
                        'cerebellum_probability_image' : 'BrainSegmentationPosteriors6.nii.gz'
                        }

# Output each session and the SST
input_image_names = [ os.path.basename(i) for i in args.head_images ]

for idx, output_basename in enumerate(input_image_names):
    # append number to session output dir
    output_file_prefix = re.split('.nii(.gz)?', output_basename)[0]
    session_output_dir = os.path.join(output_dir, f"{output_file_prefix}")

    if not os.path.exists(session_output_dir):
        os.makedirs(session_output_dir)

    for key in output_file_suffixes.keys():
        image = deep_ct[idx][key]
        output_file = os.path.join(session_output_dir, output_file_prefix + output_file_suffixes.get(key))
        print(f'Writing {key} as {output_file}')
        ants.image_write(image, output_file)

    # write transforms
    # example transforms deep_ct[idx]['template_transforms'] = {'fwdtransforms': ['/var/folders/zd/9qflmdq95ks8q89wwk9461l80000gn/T/tmpf22vu_6a0GenericAffine.mat'], 'invtransforms': ['/var/folders/zd/9qflmdq95ks8q89wwk9461l80000gn/T/tmpf22vu_6a0GenericAffine.mat']}
    # Transforms will either be {prefix}0GenericAffine.mat or {prefix}1Warp.nii.gz or both

    forward_transforms = deep_ct[idx]['template_transforms']['fwdtransforms']
    inverse_transforms = deep_ct[idx]['template_transforms']['invtransforms']

    # Parse 0GenericAffine.mat and 1Warp.nii.gz and write output to
    # session_output_dir/{output_file_prefix}SubjectToTemplate_{suffix}
    for transform in forward_transforms:
        if transform.endswith('0GenericAffine.mat'):
            output_file = os.path.join(session_output_dir, f"{output_file_prefix}SubjectToTemplate_0GenericAffine.mat")
            print(f'Writing {transform} as {output_file}')
            shutil.copyfile(transform, output_file)
        elif transform.endswith('1Warp.nii.gz'):
            output_file = os.path.join(session_output_dir, f"{output_file_prefix}SubjectToTemplate_1Warp.nii.gz")
            print(f'Writing {transform} as {output_file}')
            shutil.copyfile(transform, output_file)
        else:
            print(f'Unknown transform {transform}')


    for transform in inverse_transforms:
        # Reverse numbering for inverse transforms
        if transform.endswith('0GenericAffine.mat'):
            output_file = os.path.join(session_output_dir, f"{output_file_prefix}TemplateToSubject_1GenericAffine.mat")
            print(f'Writing {transform} as {output_file}')
            shutil.copyfile(transform, output_file)
        elif transform.endswith('1Warp.nii.gz'):
            output_file = os.path.join(session_output_dir, f"{output_file_prefix}TemplateToSubject_0Warp.nii.gz")
            print(f'Writing {transform} as {output_file}')
            shutil.copyfile(transform, output_file)
        else:
            print(f'Unknown transform {transform}')

# Now output the SST
sst_output_dir = os.path.join(output_dir, 'SingleSubjectTemplate')
if not os.path.exists(sst_output_dir):
    os.makedirs(sst_output_dir)
# SST is the last entry in deep_ct
sst = deep_ct[-1]
ants.image_write(sst, os.path.join(sst_output_dir, 'SingleSubjectTemplate.nii.gz'))

