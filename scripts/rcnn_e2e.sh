#!/bin/bash

COUNT=1

for CONTRAST in 0 0.5
do
	for BLUR in 0 0.5 1
	do
		echo Classifing $COUNT BLUR $BLUR and contrast $CONTRAST

		econvert -i feature_extraction/rcnn-region-proposals/image-354.jpg --contrast $CONTRAST --blur $BLUR -o feature_extraction/rcnn-region-proposals/e2e_trial3/image-354_$COUNT.jpg

		sleep 5

		echo ~/caffe/examples/feature_extraction/rcnn-region-proposals/e2e_trial3/image-354_$COUNT.jpg > ~/caffe/examples/_temp/det_input.txt

		../python/detect.py --crop_mode=selective_search --pretrained_model=../models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel --model_def=../models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt --gpu --raw_scale=255 _temp/det_input.txt _temp/det_output.h5

		sleep 5

		python detection.py image-354_$COUNT.jpg /home/animesh/caffe/examples/feature_extraction/rcnn-region-proposals/e2e_trial3/

		let COUNT=COUNT+1
		sleep 5
	done
done

COUNT=1
for CONTRAST in 0 0.5
do
	for BLUR in 0 0.5 1
	do
		echo $COUNT CONTRAST $CONTRAST BLUR $BLUR
		let COUNT=COUNT+1
	done
done