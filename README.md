# Deep-Atrous-CNN-NER: Word level model for Named Entity Recognition

A Deep Atrous CNN architecture suitable for Named Entity Recognition on input with variable length, which achieves state of the art results. Up to 10x times faster during prediction time.

The architecture replaces the predominant LSTM-based architectures for Named Entity Recognition tasks. Instead it uses fully convolutional model with dilated convolutions, which are resolution perserving. The architecture is inspired by the ByteNet model described in [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099).

The model has several components:
1. Embedding layer for word representation
1. 1x1 convolutions for depth decompression
1. An atrous-cnn part which is similar to the ByteNet encoder described in [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099)
1. 1x1 Convolutions for depth compression
1. A fully connected layer and SoftMax for numerical stability

Tailored cross-entropy function is applied at the end. Dropout is applied within every ResNet block in the atrous-cnn component.

<p align="center">
  <img src="https://raw.githubusercontent.com/randomrandom/deep-atrous-ner/master/png/architecture.png" width="1024"/>
</p>

The network support embedding initialization with pre-trained GloVe vectors ([GloVe: Gloval Vectors for Word Representations](https://nlp.stanford.edu/pubs/glove.pdf)) which handle even rare words quite well compared to word2vec.

To speed up training the model pre-processes any input into "clean" file, which then utilizes for training. The data is read by line from the "clean" files for better memory management. All input data is split into the appropriate buckets and dynamic padding is applied, which provides better accuracy and speed up during training. The input pipeline can read from multiple data sources which makes addition of more data sources easy as long as they are preprocessed in the right format. The model can be trained on multiple GPUs if the hardware provides this capability.

<p align="center">
  <img src="https://raw.githubusercontent.com/randomrandom/deep-atrous-ner/master/png/queue_example.gif" width="1024"/>
</p>

(Some images are cropped from [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499), [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099) and [Tensorflow's Reading Data Tutorial](https://www.tensorflow.org/programmers_guide/reading_data)) 

## Version

Current version : __***0.0.0.1***__

## Dependencies ( VERSION MUST BE MATCHED EXACTLY! )
1. numpy==1.13.0
1. pandas==0.20.2
1. protobuf==3.3.0
1. python-dateutil==2.6.0
1. scipy==0.19.1
1. six==1.10.0
1. sklearn==0.18.2
1. sugartensor==1.0.0.2
1. tensorflow==1.2.0
1. tqdm==4.14.0


## Installation
1. python3.5 -m pip install -r requirements.txt
1. install tensorflow or tensorflow-gpu, depending on whether your machine supports GPU configurations

## Dataset & Preprocessing 
Currently the only supported dataset is the [CoNLL-2003](http://www.cnts.ua.ac.be/conll2003/ner/), which can be found within the repo, additional instructions how to obtain and preprocess it can be found [here](http://www.cnts.ua.ac.be/conll2003/ner/)

The CoNLL-2003 dataset contains around 15,000 sententences (~203,000 tokens), along with a validation and test sets consisting of 3466 sentences (~51,000 tokens) and 3684 sentences (~46,400 tokens) respectively. From these tokens there are mainly 5 different entities: Person (PER), Location(LOC), Organization(ORG),
Miscellaneous (MISC) along with
non entity elements tagged as (O).

## Training the network

Before training the network you need to preprocess all the files. Do this by running `python preprocess.py`

The model can be trained across multiple GPUs to speed up the computations. In order to start the training:

Execute
<pre><code>
python train.py ( <== Use all available GPUs )
or
CUDA_VISIBLE_DEVICES=0,1 python train.py ( <== Use only GPU 0, 1 )
</code></pre>

Currently the model achieves up to 94 f1-score on the validation set.

## Monitoring and Debugging the training
In order to monitor the training, accuracy and other interesting metrics like gradients, activations, distributions, etc. across layers do the following:


```
# when in the project's root directory
bash launch_tensorboard.sh
```

then open your browser [http://localhost:6008/](http://localhost:6008/)

<p align="center">
  <img src="https://raw.githubusercontent.com/randomrandom/deep-atrous-ner/master/png/tensorboard.png" width="1024"/>
</p>

(kudos to [sugartensor](https://github.com/buriburisuri/sugartensor) for the great tf wrapper which handles all the monitoring out of the box)

During training you can also monitor the f1-scores on the validation set, which are written after every epoch on the console in a similar format:

<pre><code>
Epoch 49 - f1 scores of the meaningful classes: [ 0.93855503  0.8487395   0.93145357  0.87792642]
Epoch 49 - total f1 score: 0.9051094890510949
Epoch 50 - f1 scores of the meaningful classes: [ 0.94793926  0.86325211  0.95052474  0.88527397]
Epoch 50 - total f1 score: 0.9179461364208408
Improved F1 score, max model saved in file: asset/train/max_model.ckpt
</code></pre>

If the f1-score exceeds the previous best result a `max_model.ckpt` is saved automatically.
## Testing
You can load any previously saved `max_model.ckpt` for further testing on different datasets, by running:
<pre><code>
python test.py ( <== Use all available GPUs )
or
CUDA_VISIBLE_DEVICES=0,1 python test.py ( <== Use only GPU 0, 1 )
</code></pre>

It will produces a similar output:
```
Precision scores of the meaningful classes: [ 0.9712689   0.89571279  0.89779375  0.80020146] 
Recall scores of the meaningful classes: [ 0.956563845  0.87803279  0.92972093  0.80620112]
F1 scores of the meaningful classes: [ 0.96567506  0.88388911  0.91051454  0.80530973]
Total precision score: 0.9188072859744991  
Total recall score: 0.8992679076693969   
Total f1 score: 0.9083685761886454
```

Which in the case of the CoNLL-2003 score represents the following table:

Class | Precision | Recall | F1
--- | --- | --- | ---
*PER* | 97.22 | 95.88 | 96.56
*ORG* | 89.99 | 87.92 | 91.94
*LOC* | 95.89 | 96.99 | 96.42
*MISC* | 80.43 | 89.62 | 89.58
*Total* | 93.92 | 95.88 | 94.35

The script will run by default on the `testb` dataset.

## Future works
1. Increase the number of supported datasets
1. Put everything into Docker
1. Create a REST API for an easy deploy as a service

## My other repositories
1. [Deep-Atrous-CNN-Text-Network](https://github.com/randomrandom/deep-atrous-cnn-sentiment)

## Citation

If you find this code useful please cite me in your work:

<pre><code>
George Stoyanov. Deep-Atrous-CNN-NER. 2017. GitHub repository. https://github.com/randomrandom.
</code></pre>

## Authors
George Stoyanov (georgi.val.stoyan0v@gmail.com)
