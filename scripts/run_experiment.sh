#!/bin/bash

for FEATURIZATION in 2 3 4 5 7 8
do
	for DIMRED in PCA CCA GRP
	do
		FILE=$FEATURIZATION\_$DIMRED
		FILE_NAME=\_$FILE

		python clustering_kinematics.py $FILE_NAME\_W\_ --visual $FILE\.p
		python clustering.py $FILE\.p $FILE_NAME\_ZW\_
	done
done