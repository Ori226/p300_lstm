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
bleach==1.5.0
certifi==2016.2.28
future==0.16.0
html5lib==0.9999999
Keras==2.0.8
Mako==1.0.6
Markdown==2.6.9
MarkupSafe==1.0
nose==1.3.7
numpy==1.13.1
protobuf==3.4.0
PyYAML==3.12
scikit-learn==0.19.0
scipy==0.19.1
six==1.11.0
tensorflow==1.3.0
tensorflow-tensorboard==0.1.7
Werkzeug==0.12.2
wincertstore==0.2
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




