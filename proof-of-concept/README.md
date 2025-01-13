## Proof of Concept

A proof of concept done over the winter holidays to confirm whether this idea has any basis in reality.  
Note: to run succesfully on DCS, need to install venv in kudu. This needs the following `pip.conf` file to be added in the venv's root directory:
```
[install]
user = false
```

### Dataset

The datset used was the FaceForensics++ dataset ([Rosslet et al](https://github.com/ondyari/FaceForensics)) for the original videos with the DeepFaked videos generated through the Google and JigSaw DeepFake Detection Challenge ([Dufour et al](https://github.com/ondyari/FaceForensics)). 50 real and 50 fake videos were used for testing the proof of concepts.

### Blink detection

Blink detection was accomplished using [Google MediaPipe's Face Landmarker model](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) to detect the key points around the eyes to then calculate EAR (Eye Aspect Ratio) to determine whether a blink has occurred. The threshold for a blink was set at `min(EAR) + 0.5 * standard_deviation(EAR)` to account for the variability in the data. The result for blink detection were as follows:

True Positives (declared real when real): 44  
True Negatives (declared fake when fake): 9  
False Positives (declared real when fake): 14  
False Negatives (delcared fake when real): 5  
Unknown Real: 1  
Unknown Fake: 27  
Accuracy: 0.7361111111111112  
Accuracy (assuming unknowns are classed fake): 0.8  
Accuracy (on fake videos): 0.72  

| Confusion Matrix: | Predicted Real | Predicted Fake |
|-|-|-|
| Actual Real | 44   | 5  |
| Actual Fake | 14   | 9  |

### Conventional Detection

The Fast Gradient Sign Method was prooved effective against a VGG-based and ResNet-based DeepFake Detector [Gandhi and Jain](https://ieeexplore.ieee.org/abstract/document/9207034). I used implementations from [Yadav et al](https://github.com/rahul9903/Deepfake/blob/main/Deepfake_detection.ipynb) and [Krishna et al](https://www.kaggle.com/code/navneethkrishna23/deepfake-detection-vgg16) to train and then predict using VGG19 and [Manas Tiwari](https://www.kaggle.com/code/lightningblunt/deepfake-image-detection-using-resnet50) for the ResNet model (the resnet model was modified to output a binary output rather than a probability to unify the vgg and resnet implementations). An additional 50 real and 50 fake videos were used for testing the proof of concepts. The result for conventional detection were as follows:

**VGG19**  
True Positives: 48  
True Negatives: 50  
False Positives: 0  
False Negatives: 2  
Accuracy: 0.98  
Accuracy (on fake videos): 1

| Confusion Matrix | Predicted Real | Predicted Fake |
|-|-|-|
| Actual Real | 48  | 2  |
| Actual Fake | 0   | 50 |

**ResNet**  
True Positives: 45  
True Negatives: 45  
False Positives: 4  
False Negatives: 5  
Accuracy: 0.91  
Accuracy (on fake videos): 0.9  

| Confusion Matrix | Predicted Real | Predicted Fake |
|-|-|-|
| Actual Real | 45  | 5  |
| Actual Fake | 4   | 46 |

### Perturbation using Fast Gradient Sign Method

Initially the CW-L2 method was going to be used but was too slow. FGSM was also implemented by [Foolbox](https://github.com/bethgelab/foolbox/tree/master) and so was used as a drag and drop replacement. Work will need to be done to try and find either a faster implementation of CW-L2 or a way to speed up the process. Only fake videos had noise added to them, to refelct reliality. The results for perturbation using FGSM are as follows:

**VGG19**  
True Positives: 48  
True Negatives: 0  
False Positives: 50  
False Negatives: 2  
Accuracy: 0.52  
Accuracy (on fake videos): 0

| Confusion Matrix | Predicted Real | Predicted Fake |
|-|-|-|
| Actual Real | 48  | 2  |
| Actual Fake | 50  | 0  |

**ResNet**
True Positives: 45  
True Negatives: 50  
False Positives: 0  
False Negatives: 5  
Accuracy: 0.95  
Accuracy (on fake videos): 1

| Confusion Matrix | Predicted Real | Predicted Fake |
|-|-|-|
| Actual Real | 45  | 5  |
| Actual Fake | 0   | 50 |


**Blink Detection**  
True Positives: 44  
True Negatives: 14  
False Positives: 18  
False Negatives: 5  
Unknown Fake: 18  
Unknown Real: 1  
Accuracy: 0.716049...  
Accuracy (assuming unkown as fake): 0.76  
Accuracy (on fake videos): 0.64

| Confusion Matrix | Predicted Real | Predicted Fake |
|-|-|-|
| Actual Real | 44  | 5  |
| Actual Fake | 18  | 14 |

### Conclusion

The results are promising: adversarial perturbation can be used to fool a VGG19-based DeepFake detector, but has little to no effect on different models, thus a deliberate attack on a blink detection model will need to be developed. It is worth noting that when changing the method of adding noise, the results of blink detection remained unchanged implying an inherent resistance to noise attacks.

The next steps are to improve the blink detection model (CNNs?) and to find a faster implementation of the CW-L2 method and investigate other noise attacks. Custom implementations of all other attacks will need to be made to allow for specific targeting of the blink detection model which current methods do not provide. The project will then be expanded to include more models and more datasets to further test the robustness of the adversarial perturbation.