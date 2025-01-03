#!/bin/bash

containerVersion=""

scriptPath=$(readlink -f "$0")
scriptDir=$(dirname "${scriptPath}")

# Repo base dir under which we find bin/ and containers/
repoDir=${scriptDir%/bin}

if [[ $# -eq 0 ]]; then
    echo "
  $0 -v <antspynet version> [-B src:dest,src:dest,...] runscript [runscript_options]

Wrapper for calling ANTsPyNet, in a script or interactively.

The latest singularity (module load singularity) is loaded automatically. The call to singularity
will override \$HOME, so user runScripts should not reference \$HOME or ~/. You can if needed mount
your home directory and refer to it explicitly as /home/user.

Use absolute paths for bind points.

Script args after the runscript should reference paths within the container. For example, if
you want to use '-i FILE', FILE should be a path that is mounted at run time inside the
container with -B.

Required args:

  -v version
     ANTsPyNet version. The script will look for containers/antspynet-[version]-with-data.sif.

  runscript
    One of: t1w_segmentation (does cortical + subcortical segmentation)
            interactive
            /path/to/custom/script.py

    The run script arg can either be a reference to supported functions like t1w_segmentation, or a custom
    script. To see usage for supported functions, add the option -h.

Options:

  -B src:dest[,src:dest,...,src:dest]
     Use this to add mount points to bind inside the container, that aren't handled by other options.
     'src' is an absolute path on the local file system and 'dest' is an absolute path inside the container.
     The run script, if provided, is mounted automatically.

  -w workdir
     Initial working directory inside the container.
"
    exit 1
fi

workdir=""

while getopts "B:v:w:" opt; do
    case $opt in
        B) userBindPoints=$OPTARG;;
        v) containerVersion=$OPTARG;;
        w) workdir=$OPTARG;;
        \?) echo "Unknown option $OPTARG"; exit 2;;
        :) echo "Option $OPTARG requires an argument"; exit 2;;
    esac
done

shift $((OPTIND-1))

module load singularity

export SINGULARITYENV_ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$LSB_DJOB_NUMPROC
export SINGULARITYENV_TF_NUM_INTRAOP_THREADS=1
export SINGULARITYENV_TF_NUM_INTEROP_THREADS=1
export SINGULARITYENV_TMPDIR=/tmp

jobTmpDir=$( mktemp -d -p /scratch antspynet.${LSB_JOBID}.XXXXXXX.tmpdir ) ||
    ( echo "Could not create job temp dir ${jobTmpDir}"; exit 1 )

image="${repoDir}/containers/antspynet-${containerVersion}-with-data.sif"

scriptArg=$1
shift

singularityArgs="--cleanenv --no-home --home /home/antspyuser"

if [[ -n "${workdir}" ]]; then
  singularityArgs="$singularityArgs --pwd $workdir"
fi

if [[ -n "$userBindPoints" ]]; then
  singularityArgs="$singularityArgs \
  -B $userBindPoints"
fi

if [[ ${scriptArg,,} == "interactive" ]]; then

    # If in interactive mode, check we're in an interactive job
    if [[ ! ${LSB_INTERACTIVE} == "Y" ]]; then
        echo "interactive mode requires an interative job, eg with ibash"
        exit 1
    fi

    singularity run \
      $singularityArgs \
      $image "$@"

    exit 0
fi

# Set this to the full path to the run script
runScript=""

if [[ -f "$scriptArg" ]]; then
    runScript=$(readlink -e $scriptArg)
elif [[ -f "${repoDir}/scripts/antsxnet_${scriptArg}.py" ]]; then
    runScript=$(readlink -e "${repoDir}/scripts/antsxnet_${scriptArg}.py")
fi

if [[ ! -f "${runScript}" ]]; then
    echo "Cannot find run script $scriptArg"
    exit 1
fi

singularityArgs="$singularityArgs \
 -B ${jobTmpDir}:/tmp,${runScript}:/opt/runScript.py"

echo "
--- Container details ---"
singularity inspect $image
echo "---
"

# Command string without args to script
cmd="singularity exec \
  $singularityArgs \
  $image /opt/runScript.py"

echo "
--- Running ---
$cmd $@
---
"

# Run this way to preserve quoted args
($cmd "$@")
singExit=$?

# clean up tmp
rm -rf ${jobTmpDir}

if [[ $singExit -ne 0 ]]; then
    echo "
ERROR: Container exited with non-zero code $singExit
"
fi

exit $singExit

