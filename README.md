# pmacsANTsPyNet
Framework for running ANTsPyNet tools on the PMACS LPC.


## Basic usage

```
  runANTsXNet.sh -v <antspynet version> [-B src:dest,src:dest,...] runscript [runscript_options]
```

## Containers

Place containers inside the `containers/` directory. The script will look for
`containers/antspynet-[version]-with-data.sif`, and the user must specify the version at
run time.


## Run scripts

A custom script can be run by providing the path to the script, or use one of the
supported scripts listed below. Run with `-h` to get help, eg:
```
runANTsXNet.sh -v latest brain_extraction -h
```

The supported scripts are:

* `brain_extraction` - calls the deep learning brain extraction.
* `brain_segmentation` - calls deep Atropos, with customizable preprocessing
* `cerebellum_morphology` - cerebellum labeling
* `cortical_thickness` - calls the cortical thickness script, and saves results in a
  format similar to the `antsCorticalThickness.sh` pipeline in ANTs.
* `desikan_killiany_tourville_labeling` - cortical labeling
* `hippocampal_segmentation` - "deep flash" hippocampal segmentation
* `lesion_segmentation` - lesion segmentation on T1w
* `mri_super_resolution` - MRI super resolution, increases spatial resolution by a factor
  of two.
* `whole_head_inpainting` - inpainting of masked lesions on T1w (FLAIR to be added)
