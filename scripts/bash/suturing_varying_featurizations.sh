#!/bin/bash

cd ~/DeepMilestones/scripts

# python clustering.py SIFT_1.p _SIFT_1_v5
# python clustering.py SIFT_2.p _SIFT_2_v5

# python clustering.py 2_PCA.p _2_PCA_v5
# python clustering.py 2_CCA.p _2_CCA_v5
# python clustering.py 2_GRP.p _2_GRP_v5

# python clustering.py 3_PCA.p _3_PCA_v5
# python clustering.py 3_CCA.p _3_CCA_v5
# python clustering.py 3_GRP.p _3_GRP_v5

# python clustering.py 4_PCA.p _4_PCA_v5
# python clustering.py 4_CCA.p _4_CCA_v5
# python clustering.py 4_GRP.p _4_GRP_v5

# python clustering.py 5_PCA.p _5_PCA_v5
# python clustering.py 5_CCA.p _5_CCA_v5
# python clustering.py 5_GRP.p _5_GRP_v5

# python clustering.py 7_PCA.p _7_PCA_v5
# python clustering.py 7_CCA.p _7_CCA_v5
# python clustering.py 7_GRP.p _7_GRP_v5

python clustering_kinematics.py _kinclust_v5

python clustering_kinematics.py _visual_SIFT_1_ --visual SIFT_1.p
python clustering_kinematics.py _visual_SIFT_2_ --visual SIFT_2.p

python clustering_kinematics.py _visual_2_PCA_ --visual 2_PCA.p
python clustering_kinematics.py _visual_2_CCA_ --visual 2_CCA.p
python clustering_kinematics.py _visual_2_GRP_ --visual 2_GRP.p

python clustering_kinematics.py _visual_3_PCA_ --visual 3_PCA.p
python clustering_kinematics.py _visual_3_CCA_ --visual 3_CCA.p
python clustering_kinematics.py _visual_3_GRP_ --visual 3_GRP.p

python clustering_kinematics.py _visual_4_PCA_ --visual 4_PCA.p
python clustering_kinematics.py _visual_4_CCA_ --visual 4_CCA.p
python clustering_kinematics.py _visual_4_GRP_ --visual 4_GRP.p

python clustering_kinematics.py _visual_5_PCA_ --visual 5_PCA.p
python clustering_kinematics.py _visual_5_CCA_ --visual 5_CCA.p
python clustering_kinematics.py _visual_5_GRP_ --visual 5_GRP.p

python clustering_kinematics.py _visual_7_PCA_ --visual 7_PCA.p
python clustering_kinematics.py _visual_7_CCA_ --visual 7_CCA.p
python clustering_kinematics.py _visual_7_GRP_ --visual 7_GRP.p