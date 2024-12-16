## Proof of Concept

A proof of concept done over the winter holidays to confirm whether this idea has any basis in reality.

### Dataset

The datset used was the FaceForensics++ dataset (Rosslet et al) for the original videos with the DeepFaked videos generated through the Google and JigSaw DeepFake Detection Challenge (Dufour et al). 50 real and 50 fake videos were used for the proof of concept.

### Blink detection

Blink detection was accomplished using Google MediaPipe's Face Landmarker model to detect the key points around the eyes to then calculate EAR (Eye Aspect Ratio) to determine whether a blink has occurred. The threshold for a blink was set at min(EAR) + 0.5 * standard deviation(EAR) to account for the variability in the data. The result for blink detection were as follows:

True Positives: 44
True Negatives: 14
False Positives: 18
False Negatives: 5
Unknown Real: 1
Unknown Fake: 18
Accuracy: 0.7160493827160493