# Line Recognition for the IAM Database

### [**Contents**](#)
1. [Project Description](#descr)
1. [Setup](#setup)
2. [Data Configuration](#dataset)

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

<!---
We assume that the DSS database has been previously downloaded onto the device and is stored within a directory labeled *DSS/*. Within this directory, two sub-folders are present. The first, titled *image-data/*, houses the complete test image scripts, while the second, named *monkbrill/*, accommodates the training images. In particular, *monkbrill* comprises 27 sub-folders, each designated for a distinct Hebrew character. To proceed, copy the data directory into the project directory in the following manner:


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

-->


---



