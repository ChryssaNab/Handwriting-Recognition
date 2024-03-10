# Character Segmentation and Recognition for the <br > Dead Sea Scrolls Database

### [**Contents**](#)
1. [Project Description](#descr)
1. [Setup](#setup)
2. [Data Configuration](#dataset)
3. [Testing](#testing)
4. [Training](#training)

---

### [**Project Description**](#) <a name="descr"></a>

This project aims to recognize characters from ancient handwritten Hebrew texts, specifically those originating from the restricted collection known as the **Dead Sea Scrolls (DSS).**  Despite decay due to age, our goal is to efficiently digitize these texts. The pipeline is designed to accept images containing complete handwritten Hebrew texts from the Dead Sea Scrolls collection as input and output text documents containing the digitalized Hebrew characters present in the images.

This task involves a multi-step process: initially segmenting the textual images into lines and individual characters, followed by the recognition and transcription of these characters into *.txt* format. To achieve this, we begin by training a 2D CNN specifically designed for Hebrew characters. Once trained, we employ this model to predict the characters segmented from the test images. 



---

### [**Setup**](#) <a name="setup"></a>

**1.** We assume that Python3 is already installed on the system. The code has been specifically tested on Python version 3.10.

**2.** Clone this repository: 

``` shell
$ git clone https://github.com/ChryssaNab/Handwriting-Recognition.git
$ cd Handwriting-Recognition/character_recognition/
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

<!---
We assume that the DSS database has been previously downloaded onto the device and is stored within a directory labeled *DSS/*. Within this directory, two sub-folders are present. The first, titled *image-data/*, houses the complete test image scripts, while the second, named *monkbrill/*, accommodates the training images. In particular, *monkbrill* comprises 27 sub-folders, each designated for a distinct Hebrew character. To proceed, copy the data directory into the project directory in the following manner:
-->

We assume that the DSS database has been previously downloaded onto the device and is stored within a directory labeled *DSS/*. Inside this directory, two sub-folders are present. The first one, titled *image-data/*, contains the complete set of test image scripts. Before executing the entire pipeline for character segmentation and recognition, it is crucial to verify that the dataset adheres to the correct naming convention. This entails ensuring that all binarized file versions include the word 'binarized' in their filenames. In total, there are 20 binarized Hebrew imaging texts for testing.

Furthermore, the second sub-folder, named *monkbrill/*, accommodates the training images. In particular, *monkbrill* comprises 27 sub-folders, each designated for a distinct Hebrew character for training. The structure of the parent data folder is outlined as follows:

``` bash
DSS/
├── image-data/
    ├── P21-Fg006-R-C01-R01-binarized.jpg
    ├── P22-Fg008-R-C01-R01-binarized.jpg
    ├── P106-Fg002-R-C01-R01-binarized.jpg
    └── ...
└── monkbrill/
    ├── Alef/
        ├── navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm
        ├── navis-QIrug-Qumran_extr09_0001-line-009-y1=426-y2=543-zone-HUMAN-x=1650-y=0023-w=0035-h=0042-ybas=0045-nink=631-segm=COCOS5cocos.pgm
        └── ...
    ├── Ayin/
        ├── navis-QIrug-Qumran_extr09_0002-line-003-y1=196-y2=314-zone-HUMAN-x=0458-y=0055-w=0038-h=0032-ybas=0068-nink=562-segm=COCOS5cocos.pgm
        ├── navis-QIrug-Qumran_extr09_0002-line-004-y1=225-y2=354-zone-HUMAN-x=0458-y=0026-w=0038-h=0032-ybas=0039-nink=562-segm=COCOS5cocos.pgm
        └── ...
    ├── Bet/
        ├── navis-QIrug-Qumran_extr09_0001-line-004-y1=315-y2=423-zone-HUMAN-x=1672-y=0051-w=0033-h=0038-ybas=0049-nink=514-segm=COCOS5cocos.pgm
        ├── navis-QIrug-Qumran_extr09_0001-line-005-y1=334-y2=443-zone-HUMAN-x=1672-y=0032-w=0033-h=0039-ybas=0030-nink=518-segm=COCOS5cocos.pgm
        └── ...
    └── ...
```

To proceed, please copy the data directory into the project directory using the following command:

``` shell
$ cp -r /path/to/source_folder/DSS ./
```


---

### [**Testing**](#) <a name="testing"></a>

Execute the *main.py* script to initiate the end-to-end pipeline and set the `--data_path`, to the *image-data/* test set. This process involves segmenting the test imaging scripts into individual characters, recognizing them, and transcribing them into a text document. To facilitate this, a trained model named `model.h5`, located under the *./src/training/* directory, is loaded and utilized for predictions.

``` shell
 $ python3 src/main.py --data_path ./DSS/image-data/
 ```

The output for this procedure is stored within the directory *./results/*. It comprises two main sub-folders. The first, named *segmentation_output/*, houses the segmented lines and characters for each Hebrew text. The second, named *transcript_output/*, contains a text document for each Hebrew text, comprising the transcribed characters after they have been segmented and recognized from the original image.


---

### [**Training**](#) <a name="training"></a>

If you wish to train the model from scratch, run the following command:

``` shell
 $ python3 src/training/train.py
 ```

The data path for running the command above points to the *monkbrill/* training data and is specified in the *train.py* `lines 132-133`, as demonstrated below:

```python
# Set path to the monkbrill data
data_path = "./DSS/monkbrill/"  
 ```

The output for the training process is stored under the directory *./src/training/*. It includes the loss and accuracy curves, a summary of the model architecture, the trained model checkpoint in an HDF5/H5 file for use during testing, and the label encoder transformation in a pickle file. The latter facilitates consistent evaluation of test images through uniform label encoding. Below, we showcase the learning curves of the trained model using the default settings.

<p align="center">
    <img title="Loss curve" src="https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/character_recognition/src/training/loss.jpg" height="360" width="500"/>
     <img title="Accuracy curve" src="https://github.com/ChryssaNab/Handwriting-Recognition/blob/main/character_recognition/src/training/accuracy.jpg" height="360" width="500"/>
        
</p>


