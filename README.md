# P300 LSTM
Demonstration of the code used in "Using Recurrent Neural Networks for P300-Based Brain-Computer Interface"


## Dataset and preprocessing
The code download and use data described in:

Acqualagna, Laura, and Benjamin Blankertz. "Gaze-independent BCI-spelling using rapid serial visual presentation (RSVP)." Clinical Neurophysiology 124.5 (2013): 901-908.

The code is automatically stored and cached.

## available models:
1) LDA
2) CNN
3) lstm_small
4) lstm_big
5) lstm_cnn_small
6) lstm_cnn_big


## Requirements and Installation

The code was tested on the following environment:
* OS:windows
* Python version: 3.5
* Anaconda

packages:
```
Keras==1.2.0
numpy==1.11.3
scipy==0.18.1
scikit_learn==0.19.0
Theano==0.9.0
```





## Usage Example
```
# runnng the CNN model:
python run_multi_subject_experiment.py  -model_name CNN
```

```
# runnng the big LSTM CNN model:
python run_multi_subject_experiment.py  -model_name lstm_cnn_big
```

 ## TODO
 2) more use cases and tests
 3) better comments and documentation




