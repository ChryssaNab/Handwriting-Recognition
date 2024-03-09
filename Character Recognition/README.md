# Character Segmentation and Recognition for Dead Sea Scrolls Database

### [**Contents**](#)
1. [Project Description](#descr)
1. [Setup](#setup)
2. [Data Configuration](#dataset)
3. [Testing](#testing)
4. [Training](#training)

---

### [**Project Description**](#) <a name="descr"></a>

In this project, our goal is to recognize characters from ancient handwritten Hebrew text which has decayed due to age. For this purpose, we use a private collection named Dead Sea Scrolls (DSS). The pipeline will be able to take as input an image with handwritten Hebrew text from the Dead Sea Scrolls, and output a text document containing the digitalized Hebrew characters occurring in such a text image. 

This task involves a multi-step process: first, segmenting the textual images into lines and then individual characters; next, recognizing and transcribing these characters into a *.txt* format. To achieve this, we begin by training a 2D CNN specifically designed for Hebrew characters. Once trained, we employ this model to predict the characters segmented from the test images. 

The current project was implemented in the context of the course "Handwriting Recognition" taught by Professors Lambert Schomaker and Maruf A. Dhali at University of Groningen.



---

### [**Setup**](#) <a name="setup"></a>

**1.** We assume that Python3 is already installed on the system and the Dead Sea Scrolls (DDS) database is downloaded.

**2.** Clone this repository: 

``` shell
$ git clone https://github.com/ChryssaNab/Handwriting-Recognition.git
$ cd Handwriting-Recognition/Character\ Recognition/
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

We assume that that the DSS database has been previously downloaded onto the device and is stored within a directory labeled *DSS/*. Within this directory, two sub-folders are present. The first, titled *image-data/*, houses the complete test image scripts, while the second, named *monkbrill/*, accommodates the training images. In particular, *monkbrill* comprises 27 sub-folders, each designated for a distinct Hebrew character. To proceed, copy the data directory into the project directory in the following manner:

``` shell
$ cp -r /path/to/source_folder/DSS ./
```

---

### [**Testing**](#) <a name="testing"></a>

Execute the *main.py* script to initiate the end-to-end pipeline and set the `--data_path`, to the *image-data* test set. This process involves segmenting the test imaging scripts into individual characters, recognizing them, and transcribing them into a *.txt* format. To facilitate this, a trained model named `model.h5`, located under the *./src/training/* directory, is loaded and utilized for predictions.

``` shell
 $ python3 src/main.py --data_path 'DATA_PATH'
 ```

---

### [**Training**](#) <a name="training"></a>

If you wish to train the model from scratch, run the following command:

``` shell
 $ python3 src/training/train.py
 ```

The data path for running the command mentioned above points to the *monkbrill* training data specified in `lines 132-133`, as demonstrated below:

```python
# Set path to the monkbrill data
data_path = "./DSS/monkbrill/"  
 ```

The output for the training process is saved in the directory *./src/training/*. It includes the loss and accuracy curves, a summary of the model architecture, the trained model checkpoint in an HDF5/H5 file for use during testing, and the label encoder transformation in a pickle file. The latter facilitates consistent evaluation of test images through uniform label encoding.


