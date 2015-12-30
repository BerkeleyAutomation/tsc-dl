#!/bin/bash

# This script runs all TSC-DL experiments

cd ..

for configfile in "suturing_all.yaml"
do
	echo "[RUNNING EXPERIMENT]" $configfile
	echo $configfile > ../config/defaultconfig

	echo "[KINEMATICS - W]" $configfile
	python clustering_kinematics.py _W_

	echo "[VISUAL - Z]" $configfile
	python clustering_kinematics.py _Z_ --visual 5_PCA.p

	echo "[KINEMATICS+VISUAL - ZW]" $configfile
	python clustering.py 5_PCA.p _ZW_

done