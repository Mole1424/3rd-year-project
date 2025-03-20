## Proof of Concept
- [X] Add RESNET50 model to proof of concept
- [X] Start write up for main diss
- [X] Make demos (live view, noise visualisation, etc.)

## Main model
- [X] Research and implement custom model for eye landmark detection (cropping on eye, then some kind of pre-trained model?)
  - [X] Create and annotate dataset (email authors?)
  - [X] HRNet
  - [X] PFLD
- [X] Implement EAR analysis
  - [X] 1D CNN
  - [X] LSTM layers (bidirectional?)
  - [X] Classical Methods
  - [X] find best one and use for final model

## Noise
- [X] Code own versions of all the noise functions
  - [X] CW-L2 (Cannot do takes ~1hr30min per video (also not effective))
  - [ ] FakeRetouch
  - [X] FGSM

## Testing
- [X] Email SCRTP to ask for more space
- [X] See if anyone has got access to Meta's DFDC or try and find an email
- [ ] Look into complete deepfaked models (not changing the face with another clip, but changing the face with a generated face)
- [X] Adapt script to be generalised (`python main.py <path_to_dataset>`)
- [X] Work on test script to make multithreaded and save progress as it goes (add buffer back in to speed up testing)

## Presentation
- [X] Make presentation
- [X] Practice presentation
- [X] Present presentation (ideally will have final data by this point)

## Dissertation
- [ ] Words (a lot of them (an awful lot of them)) 
