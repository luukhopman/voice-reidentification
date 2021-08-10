# Voice Reindentification

## Objective
The goal is to build a deep learning model that is capable of identifying whether two recordings belong to the same person or not. More specifically, given a set of recordings, the model determines which of the recordings are most closely related to each other, and so creating a measure of how likely two recordings belong to the same person.

## Data
The trainings data set consists of short audio fragments of different people reading aloud. Each sequence in the data set contains only one person’s voice and lasts a second. There are several recordings for each person. Everyone's voice has specific characteristics displayed in, for example, pitch timbre or tone, which allow for computerized voice recognition.

I construct the training data in the following way: (1) for each of the 9,600 observations, I append the feature array of the 16 observations of that person to the reference features. These are the positive examples, that are labeled with one. (2) I also match the observation to audio fragments of 32 random persons and label those with zero. The resulting training data has 9,600 × (16+32) = 460,800 observations with 562 features each.

## Feature Extraction
I use the Python package LibROSA [1] to extract features the following features from the frequency waves:
- Mel-frequency cepstral coefficients (MFCCs)
- Mel-scaled spectrogram
- Chromagram of the spectrogram
- Spectral contrast
- Tonal centroid features (i.e. Tonnetz)

The following parameter values were used:
- Sampling rate: 11,025
- Fast Fourier Transform: 1,024
- Mel-frequency cepstral coefficients: 128
- Hop length: 512
- Frequency cutoff: 16

## Deep Learning Model
The trained model is a multilayer perceptron that includes 6 dense layers with LeakyReLu [2] activation and one output layer with Softmax activation. For regularization, I include batch normalization and dropout. Further, I use early stopping to prevent over-fitting. I train the model for 50 epochs with batch sizes of 64. To proxy the distance between I use the first column of the output array (i.e. the probability that the two audio samples are from different persons).

## Results
The model achieves a validation accuracy of 99.07%.


## References
1. Brian McFee, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. librosa: Audio and music signal analysis in python. In *Proceedings of the 14th python in science conference*, volume 8, pages 18–25. Citeseer, 2015

2. Xiaohu Zhang, Yuexian Zou, and Wei Shi. Dilated convolution neural network with leakyrelu for environmental sound classification. In *2017 22nd International Conference on Digital Signal Processing (DSP)*, pages 1–5. IEEE, 2017.