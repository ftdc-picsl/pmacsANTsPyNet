#!/usr/bin/env python

import argparse
import os

import ants
import antspynet
import tensorflow as tf

class RawDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass

parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter,
                    prog="t1w_brain_segmentation", add_help=False,
                    description = f'''
Wrapper for calling multi-modal deep_atropos in ANTsPyNet.


Preprocessing
-------------

Input images should be preprocessed by neck trimming and conforming to LPI orientation. All modalities should be
aligned to the T1w space. Images should not be skull-stripped. Pre-processing including N4 is done automatically.


Citation
--------

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological
and medical imaging. Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6

See the ANTsPyNet website for details of algorithm-specific citations
''')

required = parser.add_argument_group('Required arguments')
required.add_argument("--t1w-image", help="Input T1w image", type=str, required=True)
required.add_argument("-o", "--output-root", help="Full path to output including file prefix", type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("--t2w-image", help="Input T2w image", type=str, default=None)
optional.add_argument("--fa-image", help="Input FA image", type=str, default=None)
optional.add_argument("--save-posteriors", help="Save posterior probability images", action='store_true')
optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
args = parser.parse_args()

output_root = args.output_root

output_dir = os.path.dirname(output_root)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

t1w = ants.image_read(args.t1w_image)
t2w = None if args.t2w_image is None else ants.image_read(args.t2w_image)
fa = None if args.fa_image is None else ants.image_read(args.fa_image)

## Brain extraction
be = antspynet.brain_extraction(t1w, verbose=True, modality='t1threetissue')

be_seg = be['segmentation_image']
be_mask = ants.threshold_image(be_seg, 0.9, 1.1)

ants.image_write(be_mask, f"{output_root}_brain_mask.nii.gz")

ants.image_write(be_seg, f"{output_root}_head_mask.nii.gz")

## deep_atropos segmentation
output_seg = antspynet.deep_atropos([t1w, t2w, fa], do_preprocessing=True, verbose=True)

seg_output_file = f"{output_root}_brain_segmentation.nii.gz"

ants.image_write(output_seg['segmentation_image'], seg_output_file)

if args.save_posteriors:
    for i, prob_image in enumerate(output_seg['probability_images']):
        ants.image_write(prob_image, f"{output_root}_probability_{i}.nii.gz")