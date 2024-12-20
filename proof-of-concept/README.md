## Proof of Concept

A proof of concept done over the winter holidays to confirm whether this idea has any basis in reality.
note: to run succesfully on DCS, need to install venv in kudu
This needs the following `pip.conf` file to be added in the venv's root directory:
```
[install]
user = false
```

### Dataset

The datset used was the FaceForensics++ dataset (Rosslet et al) for the original videos with the DeepFaked videos generated through the Google and JigSaw DeepFake Detection Challenge (Dufour et al). 50 real and 50 fake videos were used for testing the proof of concepts.

### Blink detection

Blink detection was accomplished using Google MediaPipe's Face Landmarker model to detect the key points around the eyes to then calculate EAR (Eye Aspect Ratio) to determine whether a blink has occurred. The threshold for a blink was set at min(EAR) + 0.5 * standard deviation(EAR) to account for the variability in the data. The result for blink detection were as follows:

True Positives (declared real when real): 44  
True Negatives (declared fake when fake): 14  
False Positives (declared real when fake): 18  
False Negatives (delcared fake when real): 5  
Unknown Real: 1  
Unknown Fake: 18  
Accuracy: 0.7160493827160493
Accuracy (assuming unknowns are fake): 0.76

| Confusion Matrix: | Predicted Positive | Predicted Negative |
|-|-|-|
| Actual Positive | 44  | 18 |
| Actual Negative | 5   | 14 |

### Conventional Detection

The CW-L2 attack (Carlini and Wagner) was prooved effective against a VGG-based DeepFake Detector (Gandhi and Jain). I used an implementation from [Yadav et al](https://github.com/rahul9903/Deepfake/blob/main/Deepfake_detection.ipynb) to train and then predict using VGG19. An additional 50 real and 50 fake videos were used for testing the proof of concepts. The result for conventional detection were as follows:

True Positives: 46
True Negatives: 49
False Positives: 1
False Negatives: 4
Accuracy: 0.95

| Confusion Matrix: | Predicted Positive | Predicted Negative |
|-|-|-|
| Actual Positive | 46  | 4 |
| Actual Negative | 1   | 49 |