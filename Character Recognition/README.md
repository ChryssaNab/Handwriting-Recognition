# Character Segmentation and Recognition for Dead Sea Scrolls Database

### [**Contents**](#)
1. [Project Description](#descr)
1. [Setup](#setup)
2. [Data Configuration](#dataset)
3. [Testing](#testing)
4. [Training](#training)
5. [Output](#output)

---

### [**Project Description**](#) <a name="descr"></a>


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

We assume that that the Dead Sea Scrolls (DSS) database has been previously downloaded onto the device and is stored within a directory labeled *DSS*. Within this directory, two sub-folders are present. The first, titled *image-data*, houses the complete test image scripts, while the second, named *monkbrill*, accommodates the training images. In particular, *monkbrill* comprises 27 sub-folders, each designated for a distinct Hebrew character. To proceed, copy the data directory into the project directory in the following manner:

``` shell
$ cp -r /path/to/source_folder/DSS ./
```

---

### [**Testing**](#) <a name="testing"></a>

Execute the *main.py* script to initiate the end-to-end pipeline and set the `--data_path`, to the *image-data* test data. This process involves segmenting the test imaging scripts into individual characters, recognizing them, and transcribing them into a *.txt* format. To facilitate this, a trained model named `model.h5`, located under the *./src/training/* directory, is loaded and utilized for predictions.

``` shell
 $ python3 src/main.py --data_path 'DATA_PATH'
 ```

---

### [**Training**](#) <a name="training"></a>

If you wish to train the model from scratch, run the following command:

``` shell
 $ python3 src/training/train.py
 ```

The data path for running the command mentioned above points to the *monkbrill* training data specified in lines 132-133, as demonstrated below:

```python
# Set path to the monkbrill data
data_path = "./DSS/monkbrill/"  
 ```

---

### [**Output**](#) <a name="output"></a>
