#!/bin/bash

cd ..

echo "suturing.yaml" > ../config/defaultconfig

echo "[KINEMATICS - W]"
python tscdl.py W W

echo "[VISUAL - Z] ", SIFT_1.p
python tscdl.py Z Z_SIFT_1 --visual_feature SIFT_1.p

echo "[VISUAL - Z] ", SIFT_2.p
python tscdl.py Z Z_SIFT_2 --visual_feature SIFT_2.p

echo "[VISUAL - Z] ", 2_PCA.p
python tscdl.py Z Z_2_PCA --visual_feature 2_PCA.p

echo "[VISUAL - Z] ", 2_CCA.p
python tscdl.py Z Z_2_CCA --visual_feature 2_CCA.p

echo "[VISUAL - Z] ", 2_GRP.p
python tscdl.py Z Z_2_GRP --visual_feature 2_GRP.p

echo "[VISUAL - Z] ", 3_PCA.p
python tscdl.py Z Z_3_PCA --visual_feature 3_PCA.p

echo "[VISUAL - Z] ", 3_GRP.p
python tscdl.py Z Z_3_GRP --visual_feature 3_GRP.p

echo "[VISUAL - Z] ", 3_CCA.p
python tscdl.py Z Z_3_CCA --visual_feature 3_CCA.p

echo "[VISUAL - Z] ", 4_PCA.p
python tscdl.py Z Z_4_PCA --visual_feature 4_PCA.p

echo "[VISUAL - Z] ", 4_GRP.p
python tscdl.py Z Z_4_GRP --visual_feature 4_GRP.p

echo "[VISUAL - Z] ", 4_CCA.p
python tscdl.py Z Z_4_CCA --visual_feature 4_CCA.p

echo "[VISUAL - Z] ", 5_PCA.p
python tscdl.py Z Z_5_PCA --visual_feature 5_PCA.p

echo "[VISUAL - Z] ", 5_GRP.p
python tscdl.py Z Z_5_GRP --visual_feature 5_GRP.p

echo "[VISUAL - Z] ", 5_CCA.p
python tscdl.py Z Z_5_CCA --visual_feature 5_CCA.p

echo "[VISUAL - Z] ", 7_PCA.p
python tscdl.py Z Z_7_PCA --visual_feature 7_PCA.p

echo "[VISUAL - Z] ", 7_GRP.p
python tscdl.py Z Z_7_GRP --visual_feature 7_GRP.p

echo "[VISUAL - Z] ", 7_CCA.p
python tscdl.py Z Z_7_CCA --visual_feature 7_CCA.p


echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_SIFT_1 --visual_feature SIFT_1.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_SIFT_2 --visual_feature SIFT_2.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_2_PCA --visual_feature 2_PCA.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_2_CCA --visual_feature 2_CCA.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_2_GRP --visual_feature 2_GRP.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_3_PCA --visual_feature 3_PCA.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_3_GRP --visual_feature 3_GRP.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_3_CCA --visual_feature 3_CCA.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_4_PCA --visual_feature 4_PCA.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_4_GRP --visual_feature 4_GRP.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_4_CCA --visual_feature 4_CCA.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_5_PCA --visual_feature 5_PCA.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_5_GRP --visual_feature 5_GRP.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_5_CCA --visual_feature 5_CCA.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_7_PCA --visual_feature 7_PCA.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_7_GRP --visual_feature 7_GRP.p
echo "[KINEMATICS+VISUAL - ZW]"
python tscdl.py ZW ZW_7_CCA --visual_feature 7_CCA.p