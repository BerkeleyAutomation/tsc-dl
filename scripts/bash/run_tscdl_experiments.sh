#!/bin/bash

# This script runs all TSC-DL experiments

cd ..

for configfile in "suturing_ED.yaml"
do
    echo "[RUNNING EXPERIMENT]" $configfile
    echo $configfile > ../config/defaultconfig

    python tscdl.py W _W_last_

    python tscdl.py Z _Z_last_ --visual_feature 5_PCA.p

    python tscdl.py ZW _ZW_last_ --visual_feature 5_PCA.p

done
