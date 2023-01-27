#!/bin/bash

# Parse parameters
PARAMS=""

while (( "$#" )); do
  case "$1" in
    -r|--reconstruct)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        reconstruct=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;

    -p|--path)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        path=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;

    -m|--modes)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        modes=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;

    -u|--username)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        username=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;

    -f|--filtering)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        filtering=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;

    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;

    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;

  esac
done


# Assign default value to parameters if they are not initialized
echo
echo "################################################################################################"
if [[ -z $reconstruct ]]; then
    reconstruct="terminal"
    echo "Defaulted reconstruct to terminal, assign with e.g.: -r gui, possibilites are gui, terminal, false"
else
    echo "Reconstruct: "$reconstruct
fi

if [[ -z $username ]]; then
    username="simonne"
    echo "Defaulted username to simonne, assign with e.g.: -u tombombadil"
else
    echo "Username: "$username
fi

if [[ -z $path ]]; then
    path=$(pwd)
    echo "Defaulted path to data to cwd, assign with e.g.: -path/to/data/"
else
    echo "Path to data: "$path
fi

if [[ -z $modes ]]; then
    modes=false
    echo "Defaulted modes decomposition to false, assign with e.g.: -m true"
else
    echo "Modes: "$modes
fi

if [[ -z $filtering ]]; then
    filtering=false
    echo "Defaulted filtering to false, assign with e.g.: -f 5"
else
    echo "Filtering: "$filtering
fi

echo "################################################################################################"
echo

# Remove '='
reconstruct=${reconstruct/#=/}
username=${username/#=/}
path=${path/#=/}
modes=${modes/#=/}
filtering=${filtering/#=/}

# Old version, since the new one is kinda boggy
echo "##############################"
echo "Connecting to slurm-nice-devel"
echo "##############################"
echo
echo "Running sbatch /data/id01/inhouse/david/p9.dev/bin/job_esrf.slurm" "$reconstruct" "$username" "$path" "$modes" "$filtering"

hostname=$(hostname)
if [[ $hostname == p9-* ]]; then
  sbatch /data/id01/inhouse/david/p9.dev/bin/job_esrf.slurm "$reconstruct" "$username" "$path" "$modes" "$filtering"

else
  ssh $username@slurm-nice-devel << EOF
    sbatch /data/id01/inhouse/david/p9.dev/bin/job_esrf.slurm "$reconstruct" "$username" "$path" "$modes" "$filtering"
    exit
EOF
fi


echo
printf "You may follow the evolution of the job by typing:\n\t 'tail -f gwaihir_XXXXX.out'\n"
echo "Replace XXXXX by the previous job number."
echo "The job file should be in your home directory or in $path"