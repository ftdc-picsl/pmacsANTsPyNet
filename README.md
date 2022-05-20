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
* `cortical_thickness` - calls the cortical thickness script, and saves results in a
  format similar to the `antsCorticalThickness.sh` pipeline in ANTs.

Longitudinal cortical thickness is not supported but will be in the future.