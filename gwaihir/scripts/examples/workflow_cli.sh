#/usr/bin/bash
# In this example, we have
#        Arg 1: Original data directory 
#        Arg 2: Path to target directory (e.g. Pt_Al2O3/) 
#				(subdirectories will be updated/created)
#        Arg 3: Scan(s) number, list or single value

echo "##################################################################"
echo "Remember to source your virtual environment"
echo "##################################################################"

echo "##################################################################"
echo "Moving scan to new directory and touching new directories..."
echo run_movetodir.py $1 $2 $3
echo "##################################################################"
run_movetodir.py $1 $2 $3

# echo "##################################################################"
# echo "Rotating scan..."
# echo run_rotate.py $2 $3
# echo "##################################################################"
# rotate.py $2 $3

# echo "##################################################################"
# echo "Correcting angles..."
# echo run_correct_angles_detector.py $2 $3
# echo "##################################################################"
# correct_angles_detector.py $2 $3

echo "##################################################################"
echo "Preprocessing scan..."
echo bcdi_preprocess_BCDI.py --config config_preprocessing.yml
echo "##################################################################"
bcdi_preprocess_BCDI.py --config config_preprocessing.yml \
	--save_dir $2$3/preprocessing \
	--root_folder $2$3/data \
	--scans $3

# echo "##################################################################"
# echo "Ready to launch phase retrieval !"
# echo run_slurm_job.sh --reconstruct gui --username simonne --path $2S$3/preprocessing --filtering 20 --modes true
# echo "##################################################################"

# echo "##################################################################"
# echo "Once phase retrieval is finished and you are satisfied on the final solution, you can run the postprocessing..."
# echo bcdi_strain_BCDI.py -save_dir $2S$3/preprocessing -data_dir $2S$3/data -scans $3 -config_file config_postprocessing_sixs.yml -reconstruction_file path\to\solution
# echo "##################################################################"
