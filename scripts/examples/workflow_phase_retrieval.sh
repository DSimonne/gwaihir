#usr/bin/bash

## GUI

cwd=$(pwd)
ssh $1@slurm-nice-devel << EOF

	sbatch /data/id01/inhouse/david/Packages/phdutils/phdutils/bcdi/terminal_scripts/pynx_GUI.slurm $2
    
    echo "Phase retrieval is running ..."
    
	exit

EOF


## slurm

# cwd=$(pwd)
# ssh simonne@slurm-nice-devel << EOF

# 	sbatch pynx_ID01.slurm $cwd $1 $2
    
#     echo "Phase retrieval is running ..."
    
# 	exit

# EOF

# echo "Will also run strain analysis"
# echo strain_ID01.py $1 $2

# echo "If you have the conjugated object run:"
# echo strain_ID01.py $1 $2 flip=True