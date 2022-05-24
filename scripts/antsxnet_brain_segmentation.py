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
Wrapper for deep_atropos in ANTsPyNet.

For more details of the algorithm and available options, see

https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/deep_atropos.py

Preprocessing
-------------

Input images should be preprocessed. By default, this is performed by the built-in preprocessing
function. Advanced users can turn this off, but should brain extract, bias-correct, and denoise the
input images. Affine alignment to a template is required and will always be done internally,
regardless of the "--preprocess" option.

Citation
--------

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological
and medical imaging. Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6

'''
                                )
required = parser.add_argument_group('Required arguments')
required.add_argument("-a", "--head-image", help="Input T1w image, not skull-stripped", type=str, required=True)
required.add_argument("-o", "--output-root", help="Full path to output including file prefix", type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
optional.add_argument("-m", "--brain-mask", help="T1w brain mask. Only used if preprocessing is disabled", type=str)
optional.add_argument("--no-preprocess", help="Skip internal preprocessing, including N4 and denoising. Brain extraction "
                      "will still be performed unless a brain mask is specified.",
                    action='store_true')
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

# The results to write to disk
output_seg = None

if (args.no_preprocess):
    print("--- Running deep_atropos without preprocessing ---")
    # Align input to croppedMNI152
    template_file_name_path = antspynet.get_antsxnet_data("croppedMni152")
    template_image = ants.image_read(template_file_name_path)

    if (args.brain_mask is None):
        print("--- Computing brain mask ---")
        probability_mask = antspynet.brain_extraction(anat, modality="t1", verbose=True)
        mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
        mask = mask.iMath_fill_holes()
    else:
        mask = ants.image_read(args.brain_mask)

    print("--- Computing template brain mask ---")
    template_probability_mask = antspynet.brain_extraction(template_image, modality="t1", verbose=False)
    template_mask = ants.threshold_image(template_probability_mask, 0.5, 1, 1, 0)
    template_mask = template_mask.iMath_fill_holes()
    template_brain_image = template_image * template_mask

    preprocessed_brain_image = anat * mask

    print("--- Registering brain to template ---")
    registration = ants.registration(fixed=template_brain_image, moving=preprocessed_brain_image,
                type_of_transform="antsRegistrationSyNQuickRepro[a]", verbose=True)

    preprocessed_template_space = ants.apply_transforms(fixed = template_image, moving = preprocessed_brain_image,
                           transformlist=registration['fwdtransforms'], interpolator="linear", verbose=True)

    print("--- Deep Atropos segmentation ---")
    deep_seg = antspynet.deep_atropos(preprocessed_template_space, do_preprocessing=False, verbose=True)

    # Warp posteriors back to native space
    native_probability_images = list()

    for idx in range(len(deep_seg['probability_images'])):
        native_probability_images.append(ants.apply_transforms(fixed = anat, moving = deep_seg['probability_images'][idx],
                                                               transformlist = registration['fwdtransforms'], whichtoinvert=[True],
                                                               interpolator="linear", verbose=True))

    # Make segmentation image in native space by finding the max probability in each voxel

    image_matrix = ants.image_list_to_matrix(native_probability_images, anat * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), anat * 0 + 1)[0]

    # populate output list
    output_seg = {'segmentation_image' : segmentation_image,
                  'probability_images' : native_probability_images}
else:
    print("Running deep_atropos with built-in preprocessing")
    output_seg = antspynet.deep_atropos(anat, do_preprocessing=True, verbose=True)


# Write output to disk
seg_output_file = f'{args.output_root}BrainSegmentation.nii.gz'

ants.image_write(output_seg['segmentation_image'], seg_output_file)

for idx, image in enumerate(output_seg['probability_images']):
    output_file = f'{args.output_root}BrainSegmentationPosteriors{idx}.nii.gz'
    ants.image_write(image, output_file)
