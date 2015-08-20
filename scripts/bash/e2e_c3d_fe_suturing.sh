#!/bin/bash

# This script converts all suturing .avi video files to .mat files which contain the c3d features
FILES=/home/animesh/jigsaws/Suturing_video/video/*

COUNT=1
for f in $FILES; do
	# Make all necessary directories
	echo "-------------[DEBUG] $COUNT--------------"
	echo " Processing $f file......"
	cd /home/animesh/C3D/examples/c3d_feature_extraction/output/c3d/
	mkdir $COUNT
	cd /home/animesh/C3D/examples/c3d_feature_extraction/input/frm/
	mkdir $COUNT
	cd $COUNT
	cp $f .

	# Convert video to jpg images
	ffmpeg -i $f -r 16 %06d.jpg

	cd /home/animesh/C3D/examples/c3d_feature_extraction/script/
	echo "-------------[DEBUG]--------------"
	echo "Type in number of 16-frame batches for file $f, followed by [ENTER]:"
	read NUM_BATCHES
	python generate_input_output_files.py input_list_frm_suturing.txt $COUNT output_list_prefix_suturing.txt $COUNT $NUM_BATCHES
	cd ..
	sh c3d_suturing_feature_extraction.sh

	# Convert feature file type to .mat files 
	FEATURE_FILES=/home/animesh/C3D/examples/c3d_feature_extraction/output/c3d/$COUNT/*
	cd /home/animesh/C3D/examples/c3d_feature_extraction/script
	for file in $FEATURE_FILES; do
		mkdir $COUNT
		matlab -nosplash -nodesktop -nojvm -r "read_binary_blob('$file', '$COUNT');exit"
	done
	COUNT=$((COUNT+1))
done