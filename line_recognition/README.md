# Line Recognition for the IAM Database

### [**Contents**](#)
1. [Project Description](#descr)
2. [Setup](#setup)
3. [Data Configuration](#dataset)
4. [Methodology Overview](#methodology)
5. [Execution](#execution)
6. [Inference](#inference)

---

### [**Project Description**](#) <a name="descr"></a>
This project aims to perform line recognition on a subset of the **IAM database**, which comprises handwritten English sentences. Our subset includes 7458 text images, each accompanied by its corresponding transcription label. With access to complete sentence labels, we can implement an end-to-end system for line recognition. To achieve this, we adopt the model proposed by Scheidl (2018) in his PhD thesis [[1]](#1). This model consists of a CNN followed by bidirectional LSTM layers. These layers are fed with image features in the form of time-series data. Lastly, a softmax output layer is appended to the model. This layer enables the calculation of the Connectionist Temporal Classification (CTC) loss [[2]](#2), which assists in determining the gradients.


---

### [**Setup**](#) <a name="setup"></a>

**1.** We assume that Python3 is already installed on the system. The code has been specifically tested on Python version 3.10.

**2.** Clone this repository: 

``` shell
$ git clone https://github.com/ChryssaNab/Handwriting-Recognition.git
$ cd Handwriting-Recognition/line_recognition/
```

 **3.** Create a new Python environment and activate it:
 
``` shell
$ python3 -m venv env
$ source env/bin/activate
```

**4.** Install necessary requirements:

``` shell
$ pip install wheel
$ pip install -r requirements.txt
```

---


### [**Data Configuration**](#) <a name="dataset"></a>

We assume that the IAM database has been previously downloaded onto the device and is stored within a directory labeled *IAM-data/*. Inside this directory, there is one sub-folder and one text document. The sub-folder, titled *img/*, contains 7458 images with handwritten English text lines used for training, validation, and evaluation of the model. The text document, named *iam_lines_gt.txt*, contains the ground-truth labels for each image, providing the transcribed texts. The structure of the parent data folder is outlined as follows:

``` bash
IAM-data/
├── img/
    ├── a01-000u-00.png
    ├── a01-000u-01.png
    ├── a01-000u-02.png
    └── ...
└── iam_lines_gt.txt

```

To proceed, please copy the data directory into the project directory using the following command:

``` shell
$ cp -r /path/to/source_folder/IAM-data ./
```

The data was divided into three subsets: train, validation, and test sets. The test set comprises the last 20% of our dataset, while the remaining data is shuffled. An additional 20% is then removed for validation. This leaves us with a training set that accounts for 64% of the initial dataset.

---

### [**Methodology Overview**](#) <a name="methodology"></a>

The complete workflow of our line recognition system is illustrated below. The blue pathway depicts the training process, whereas the red pathway represents the testing process.

<p align="center">
    <img title="Methodology overview" src="https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/line_recognition/imgs/pipeline.png" height="340" width="545"/>
        
</p>


---

### [**Execution**](#) <a name="execution"></a>
The primary execution script for the entire project is the *main.py* script within the *src/* directory. To view usage information run the following command:

``` shell
$ python3 src/main.py -h

usage: main.py [-h] [--mode {train,test}] [--debug] [--final] [path]

args for IAM main

positional arguments:
  path                 path to 'img' (test) or 'IAM-data' (train)

options:
  -h, --help           show this help message and exit
  --mode {train,test}  train or test model (default test)
  --debug              use debug mode (default False)
  --final              train model on the whole data set (default False)

```

The possible arguments for configuring and training the LSTM model are specified within the *settings.py* script. To train, validate, and test the model on the IAM dataset run the following command:

``` python
$ python3 src/main.py  ./IAM-data/ --mode train
```

<!--
This command loads the *IAM-data*, preprocesses it, splits it into train, validation, and test sets, and performs line recognition using the provided ground-truth labels. The output of this procedure is stored in the directory *./results/$current_date/*. It includes the trained model checkpoint from the last epoch in a `.h5` file, logs of the training and validation processes, and the training settings in a `.json` file. Additionally, two prevalent error measures in HTR, the *Character Error Rate (CER)* and the *Word Error Rate (WER)*, are printed in the command-line for each epoch for the test set.
-->

This command loads the *IAM-data*, preprocesses it, splits it into train, validation, and test sets, and conducts end-to-end line recognition using the provided ground-truth labels. The output of this procedure is stored under the directory *./results/$current_date/*. It comprises three primary sub-folders:

 - The first, named *logs/*, holds the logs of the training and validation processes, along with a text document named *output.txt* containing the model's predictions and ground-truth labels for each image.
 - The second, titled *model/*, stores the model checkpoint from the last epoch as *LSTM_model.h5*.
 - The third, labeled *settings/*, encompasses all the settings and configurations utilized during the training process.
   
Moreover, two common error metrics in HTR, the *Character Error Rate (CER)* and the *Word Error Rate (WER)*, are printed in the command-line output for each epoch for the test set. Below, we showcase the learning curves of the trained model on the validation set using the default settings.

<p align="center">
    <img title="Loss curve" src="https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/line_recognition/imgs/best_model_val_cer.png" height="360" width="500"/>
     <img title="Accuracy curve" src="https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/line_recognition/imgs/best_model_val_wer.png" height="360" width="500"/>
        
</p>


---

### [**Inference**](#) <a name="inference"></a>


**1.**  Train the model on the whole IAM dataset, setting the `--final` argument to `True`:

``` python
$ python3 src/main.py  ./IAM-data/ --mode train --final
```

**2.** Transfer the desired model and training settings folders to the directory path "*./results/run_final_model/*". Make sure to update the */$current_date/* accordingly.

``` shell
$ mkdir -p ./results/run_final_model/
$ cp -r ./results/$current_date/model ./results/run_final_model/
$ cp -r ./results/$current_date/settings ./results/run_final_model/
```

**3.** Run the *main.py* script, setting the `--mode` argument to `test`:

``` python
$ python3 src/main.py  ./IAM-data/ --mode test
```

---

### References 

<a id="1">[1]</a> 
Harald Scheidl. Handwritten text recognition in historical documents. PhD thesis, Wien, 2018.

<a id="2">[2]</a> 
Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. 2006. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. *In Proceedings of the 23rd international conference on Machine learning (ICML '06). Association for Computing Machinery, New York, NY, USA, 369–376. https://doi.org/10.1145/1143844.1143891*.
