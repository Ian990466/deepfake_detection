# Deepfake Detection

Forged videos are commonly spread online. Most have malicious content and cause serious information security problems. The most critical issue in deepfake detection is the identification of traces of tampering in fake videos. This study designs a Dual Attention Forgery Detection Network (DAFDN), which embeds a spatial reduction attention block (SRAB) and a forgery feature attention module (FFAM) to the backbone network. DAFDN embeds the two proposed attention mechanisms and enables the convolution neural network to extract peculiar traces left by images’ warping. This study uses two benchmark datasets, DFDC and FaceForensics++, to compare the performance of the proposed DAFDN with other methods. The results show that the proposed DAFDN mechanism achieves AUC scores of 0.911 and 0.945 in the datasets DFDC and FaceForensics++, respectively. These results are better than those of previously developed methods, such as XceptionNet and EfficientNet-related methods.

## Heatmaps

From top to bottom are extracted head image, own method, efficientNet without attention, xceptionnet.

<img src="./Heatmap_Result/compare.png" width= "500px"></img>