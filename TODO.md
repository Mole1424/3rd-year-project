## Proof of Concept
- [ ] Add RESNET50 model to proof of concept
- [ ] Start write up for main diss

## Main model
- [ ] Research and implement custom model for eye landmark detection (cropping on eye, then some kind of pre-trained model?)
- [ ] Implement EAR analysis (look at feature vectors being: (length of down, length of up, period closed, period between blinks, etc.))

## Noise
- [ ] Test speed of other implementations of CW-L2 attack (look at speedup with multiple-threading and GPU (not in Python?))
- [ ] Look at other methods (FGSM, FakeRetouch) and implement/improve when necessary

## Testing
- [ ] Email SCRTP to ask for more space
- [ ] See if anyone has got access to Meta's DFDC or try and find an email
- [ ] Look into complete deepfaked models (not changing the face with another clip, but changing the face with a generated face)
- [ ] Adapt script to be generalised (`python test.py <path_to_dataset>`)
- [ ] Work on test script to make multithreaded and save progress as it goes (add buffer back in to speed up testing)

## Dissertation
- [ ] Words (a lot of them (an awful lot of them)) 