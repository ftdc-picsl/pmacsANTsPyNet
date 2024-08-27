#!/usr/bin/env python

import argparse
import sys

import os.path
from os import path

import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 prog='tids_qa', add_help = False, description='''
Wrapper for tid_neural_image_assessment in AntsPyNet.

https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/quality_assessment.py

This script takes a list of images and outputs a CSV file with columns image,mean_score,sd_score
containing the global estimates of image quality.

If you use this in your research, please cite:

Tustison, N.J., Cook, P.A., Holbrook, A.J. et al. The ANTsX ecosystem for quantitative biological and medical imaging.
Sci Rep 11, 9068 (2021). https://doi.org/10.1038/s41598-021-87564-6
''')
required = parser.add_argument_group('Required arguments')
required.add_argument('-i', '--image-list', help='List of input images, where each line is either {image} ' \
                            'or {image},{mask}', type=str, required=True)
required.add_argument('-o', '--output-file', help='Output file', type=str, required=True)
optional = parser.add_argument_group('Optional arguments')
# dimensions to predict defaults to 0 but can be a vector eg 0 1 2
optional.add_argument('--dimensions-to-predict', help='Dimensions to predict. Default is 0', type=int, nargs='+', default=[0])
optional.add_argument('--patch-size', help='Patch size for quality assessment. Suggested value is 101 or global.' \
                                            'Default is global', type=str, default='global')
optional.add_argument('--no-reconstruction', help='Turn this on if you just want the predicted values (saves time)',
                      action='store_true')
optional.add_argument('-h', '--help', action='help', help='show this help message and exit')
optional.add_argument('-t', '--threads', help='Number of threads in tensorflow operations. Use environment variable ' \
                    'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS to control threading in ANTs calls', type=int, default=1
                    )
args = parser.parse_args()


# Now fire up ants etc
import ants
import antspynet
import tensorflow as tf

# Internal variables for args
image_list = args.image_list
output_file = args.output_file
threads = args.threads

tf.keras.backend.clear_session()
tf.config.threading.set_intra_op_parallelism_threads(threads)
tf.config.threading.set_inter_op_parallelism_threads(threads)

# Read in the image list
with open(image_list, 'r') as f:
    images_and_masks = f.readlines()

# Remove whitespace characters
images_and_masks = [x.strip() for x in images_and_masks]

masks = None

# Check if the format is {image},{mask}
if any(',' in s for s in images_and_masks):
    images = [x.split(',')[0] for x in images_and_masks]
    masks = [x.split(',')[1] for x in images_and_masks]
else:
    images = images_and_masks

data_rows = []  # List to store row data

patch_size = args.patch_size

# convert patch_size to int if it is a number
if patch_size.isnumeric():
    patch_size = int(patch_size)

# Now run the quality assessment
for i in range(len(images)):
    print('Processing image ' + str(i+1) + ' of ' + str(len(images)) + ': ' + images[i])
    image = ants.image_read(images[i])
    mask = None
    if masks is not None:
        mask = ants.image_read(masks[i])
    qa = antspynet.tid_neural_image_assessment(image, mask=mask, which_model='tidsQualityAssessment',
                                               patch_size=patch_size, dimensions_to_predict=args.dimensions_to_predict,
                                               no_reconstruction=args.no_reconstruction)
    row_data = {'image': images[i], 'mean_score': qa['MOS.mean'], 'sd_score': qa['MOS.standardDeviationMean']}
    data_rows.append(row_data)
# Write out the dataframe
out_df = pd.DataFrame(data_rows)
pd.DataFrame.to_csv(out_df, output_file, index=False)
