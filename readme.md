For visual + kinematics:

python clustering.py 4_PCA.p _4_PCA__

For kinematics:

python clustering_kinematics.py _kin_

For visual:

python clustering_kinematics.py _viz_ --visual 4_PCA.p


Note:
Specify the list of demonstrations in clustering.py (line 750+), clustering_kinematics.py (line 530+)

list_of_demonstrations = ["0100_01", "0100_02", "0100_03", "0100_04", "0100_05"]


List of Major Params: 
1. GMM for CP