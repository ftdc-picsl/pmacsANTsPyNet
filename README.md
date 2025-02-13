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
runANTsXNet.sh -v latest t1w_segmentation -h
```

The supported scripts are:

* `t1w_segmentation` - calls t1w brain extraction and segmentation for cortial + subcortical structures

