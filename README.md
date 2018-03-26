# Speech-to-Text-WaveNet : End-to-end sentence level Chinese speech recognition using DeepMind's WaveNet
A tensorflow implementation for Chinese speech recognition based on DeepMind's WaveNet: A Generative Model for Raw Audio.([Related Paper](https://arxiv.org/pdf/1609.03499.pdf))

My practice is based on THCHS-30. Because there is not enough time to optimize the parameter of the model. At present, the recognition accuracy is not high. This repository will continue to update as a record.

## Version
Current version: 0.0.1(Draft)
## Prerequisites
- python 3.5
- tensorflow 1.0.0
- librosa 0.5.0
## Dataset
[THCHS-30](http://www.openslr.org/18/)
## Directories
- cache: save data feature and word dictionary
- data: wav files and related labels
- model: save the models
## Network model
- Data random shuffle per epoch
- Xavier initialization
- Adam optimization algorithms
- Batch Normalization
## Train the network
python3 train.py
## Test the network
python3 test.py
## References
- [TensorFlow练习15: 中文语音识别](http://blog.topspeedsnail.com/archives/10696#more-10696)
- [ibab's WaveNet(speech synthesis) tensorflow implementation](https://github.com/ibab/tensorflow-wavenet)
- [buriburisuri's WaveNet(English speech recognition) tensorflow and sugartensor implementation](https://github.com/buriburisuri/speech-to-text-wavenet#version)
