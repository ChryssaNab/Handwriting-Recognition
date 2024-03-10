# Line Recognition for the IAM Database

### [**Contents**](#)
1. [Project Description](#descr)
2. [Setup](#setup)
3. [Data Configuration](#dataset)
4. [Methodology Overview](#methodology)

---

### [**Project Description**](#) <a name="descr"></a>



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
    <img title="Methodology overview" src="https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/line_recognition/imgs/pipeline.pdf" height="340" width="540"/>
        
</p>

---

