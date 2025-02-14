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
Wrapper for calling various brain segmentation routines in ANTsPyNet.


Preprocessing
-------------

Input images should be preprocessed by neck trimming and conforming to LPI orientation.


Citation
--------

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological
and medical imaging. Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6

See the ANTsPyNet website for details of algorithm-specific citations
''')

required = parser.add_argument_group('Required arguments')
required.add_argument("-a", "--head-image", help="Input T1w image, not skull-stripped", type=str, required=True)
required.add_argument("-o", "--output-root", help="Full path to output including file prefix", type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
args = parser.parse_args()

output_root = args.output_root

output_dir = os.path.dirname(output_root)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

anat = ants.image_read(args.head_image)

## Brain extraction
be = antspynet.brain_extraction(anat, verbose=True, modality='t1threetissue')

be_seg = be['segmentation_image']
be_mask = ants.threshold_image(be_seg, 0.9, 1.1)

ants.image_write(be_mask, f"{output_root}_brain_mask.nii.gz")

ants.image_write(be_seg, f"{output_root}_head_mask.nii.gz")

## deep_atropos segmentation
output_seg = antspynet.deep_atropos(anat, do_preprocessing=True, verbose=True)

seg_output_file = f"{output_root}_brain_segmentation.nii.gz"

ants.image_write(output_seg['segmentation_image'], seg_output_file)

# Segmentation probabilities, only really needed for use in other tools
# for idx, image in enumerate(output_seg['probability_images']):
#    output_file = f"{output_root}_brain_segmentation_posterior_{idx}.nii.gz"
#    ants.image_write(image, output_file)


## Whole hippocampal segmentation - can use HOA for hippo labels
# hipp = antspynet.hippmapp3r_segmentation(anat, verbose=True)
# ants.image_write(hipp, f"{output_root}_hippocampus_segmentation.nii.gz")

## Claustrum
# Quite inconsistent results
# claustrum_prob = antspynet.claustrum_segmentation(anat, verbose=True)
# claustrum_seg = ants.threshold_image(claustrum_prob, 0.5, 1.1)
# ants.image_write(claustrum_seg, f"{output_root}_claustrum_segmentation.nii.gz")

## Hypothalamus
# Does subfield segmentation - controversial on T1w
# hypo_seg = antspynet.hypothalamus_segmentation(anat, verbose=True)
# ants.image_write(hypo_seg['segmentation_image'], f"{output_root}_hypothalamus_segmentation.nii.gz")


## Deep flash
# Does subfield segmentation - controversial on T1w
# df = antspynet.deep_flash(anat, verbose=True)
# ants.image_write(df['segmentation_image'], f"{output_root}_deep_flash_segmentation.nii.gz")


## Harvard Oxford subcortical atlas
hoa = antspynet.harvard_oxford_atlas_labeling(anat, verbose=True)
ants.image_write(hoa['segmentation_image'], f"{output_root}_harvard_oxford_subcortical_segmentation.nii.gz")

## DKT
dkt = antspynet.desikan_killiany_tourville_labeling(anat, do_lobar_parcellation=True, version=1, verbose=True)
ants.image_write(dkt['segmentation_image'], f"{output_root}_desikan_killiany_tourville_segmentation.nii.gz")
ants.image_write(dkt['lobar_parcellation'], f"{output_root}_desikan_killiany_tourville_lobar_segmentation.nii.gz")

## Cerebellum extraction
# Use HOA mask for cerebellum extraction, seems to be more accurate than atropos
# You can also use registration to a template to get the mask, but it's slower
cerebellum_mask = ants.threshold_image(hoa['segmentation_image'], 28.5, 32.5)

cerebellum = antspynet.cerebellum_morphology(anat, cerebellum_mask=cerebellum_mask, compute_thickness_image=False,
                                             verbose=True)
ants.image_write(cerebellum['cerebellum_probability_image'], f"{output_root}_cerebellum_probability_mask.nii.gz")
ants.image_write(cerebellum['parcellation_segmentation_image'], f"{output_root}_cerebellum_parcellation.nii.gz")
