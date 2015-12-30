#!/bin/bash

# Experiment 1: Script loops through all layers in the C3D Net to extract features
COUNT=1

for LAYER in conv2a conv3a conv3b conv4a conv4b conv5a conv5b fc6-1 fc7-1 fc8-1
do
	# First make all the folders
	echo Extrating features - $LAYER
	cd /home/animesh/C3D/examples/c3d_feature_extraction/output/c3d/experiment1/
	mkdir $LAYER
	cd /home/animesh/C3D/examples/c3d_feature_extraction/script/data/experiment1/
	mkdir $LAYER

	# Generate config files needed for Feature extraction
	cd /home/animesh/C3D/examples/c3d_feature_extraction/script/
	python generate_input_output_files.py input_list_frm_suturing.txt Suturing_E005_capture2 output_list_prefix_suturing.txt experiment1/$LAYER 0 16 --segments data/Suturing_E005_capture2/Suturing_E005_capture2.p

	# Extract features from C3D
	cd /home/animesh/C3D/examples/c3d_feature_extraction/
	GLOG_logtosterr=1 ../../build/tools/extract_image_features.bin prototxt/c3d_suturing_feature_extractor_frm.prototxt conv3d_deepnetA_sport1m_iter_1900000 -1 100 1 prototxt/output_list_prefix_suturing.txt $LAYER
	cp /home/animesh/C3D/examples/c3d_feature_extraction/output/c3d/experiment1/$LAYER/* script/data/experiment1/$LAYER/
	sleep 5

	echo Plotting - $LAYER

	#Plot PCA of features
	cd script
	
	python tsne.py sutE5_cap2 $LAYER data/experiment1/$LAYER/ --a data/Suturing_E005_capture2/Suturing_E005_capture2.p
	COUNT=COUNT+1
done