# Part-Level Convolutional Neural Networks for Pedestrian Detection Using Saliency and Boundary Box Alignment


Abstract
--------
Pedestrians in videos have a wide range of appearance factors such as body poses, occlusions, and complex backgrounds, which make their detection difficult. Moreover, a proposal shift problem causes the loss of body parts such as head and legs in pedestrian detection, which further degrades the detection accuracy. In this paper, we propose part-level convolutional neural networks (CNNs) for pedestrian detection using saliency and boundary box (BB) alignment. The proposed network consists of two subnetworks: detection and alignment. In the detection subnetwork, we use saliency to remove false positives such as lamp posts and trees by combining fully convolutional network (FCN) and class activation map (CAM) to extract deep features. Subsequently, we adopt the BB alignment on detection proposals in the alignment subnetwork to overcome the proposal shift problem by applying the part-level CNN to recall the lost body parts. Experimental results on various datasets demonstrate that the proposed method remarkably improves
accuracy in pedestrian detection and outperforms existing state-of-the-art techniques.


Contribution Highlights
-----------------------

- We use saliency in the detection subnetwork to remove background components such as lamp posts and trees from pedestrians.
- We combine FCN and CAM into the alignment subnetwork to enhance the resolution of confidence maps and successfully recall the lost body parts.

