## Proof of Concept
- [X] Add RESNET50 model to proof of concept
- [X] Start write up for main diss
- [X] Make demos (live view, noise visualisation, etc.)

## Main model
- [X] Research and implement custom model for eye landmark detection (cropping on eye, then some kind of pre-trained model?)
  - [X] Model is from https://www.sciencedirect.com/science/article/pii/S0031320319303772
  - [X] Create and annotate dataset (email authors?)
  - [X] Train model
  - [ ] Test model
- [ ] Implement EAR analysis
  - [ ] 1D CNN
  - [ ] LSTM layers (bidirectional?)
  - [ ] Split blinks into feature vector then train
  - find best one and use for final model

## Noise
- [ ] Code own versions of all the noise functions
  - [ ] CW-L2
  - [ ] FakeRetouch
  - [ ] FGSM?

## Testing
- [ ] Email SCRTP to ask for more space
- [ ] See if anyone has got access to Meta's DFDC or try and find an email
- [ ] Look into complete deepfaked models (not changing the face with another clip, but changing the face with a generated face)
- [ ] Adapt script to be generalised (`python test.py <path_to_dataset>`)
- [ ] Work on test script to make multithreaded and save progress as it goes (add buffer back in to speed up testing)

## Presentation
- [ ] Make presentation
- [ ] Practice presentation
- [ ] Present presentation (ideally will have final data by this point)

## Dissertation
- [ ] Words (a lot of them (an awful lot of them)) 