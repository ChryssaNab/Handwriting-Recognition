**Executables**

  - main.py - the main executable

|-------------------------------------------------------------------------------------------------------------------------------------|

**Modules**

  - preprocessing.py - module that preprocesses binarized input image.
  - segmentation.py - module that segments preprocessed image into characters.
  - predict.py - module that recognizes segmented characters.
  - transcript.py - module that transcripts the recognized characters of an input image in a .txt file.
  
  - training/dataAugmentation.py - module that augments the images used for the training.
  - training/train.py - module that builds and trains a convolutional neural network. 
  - n_grams.py - module that implements a bigram model of Hebrew characters.
  
  - opts.py - module that contains the path settings.
  - utils.py - module that contains utility functions.

|-------------------------------------------------------------------------------------------------------------------------------------|

**Output**

  - training/model.h5 - the trained model is already saved and is loaded in the **main.py** script
  - training/LabelEncoder.pickle - the LabelEncoder is already saved and is loaded in the **predict.py** script for the recognition
  
  - ./results/segmentation_output - the folder with the segmented characters for each input image (to be created in the main.py script)
  - ./results/transcript_output - the folder with the transcription of each input image (to be created in the main.py script)

|-------------------------------------------------------------------------------------------------------------------------------------|

**Warnings**

 1. The provided input images ended-up with "binarized.jpg" among RGB and gray-scale images. In our 'main.py' script they are
    extracted and loaded in this way (line 30).
 
 2. In 'prediction.py', we import pickle to load the LabelEncoder as "import pickle5 as pickle". The debug option is "import pickle". 

|-------------------------------------------------------------------------------------------------------------------------------------|

**Commands** 

### Step 1: We assume that Python3 is already installed on the device and the Dead Sea Scrolls (DSS) dataset is downloaded.

|-------------------------------------------------------------------------------------------------------------------------------------|

### Step 2: Navigate to the project's directory.

	$ cd Handwriting_Recognition-RUG/Character\ Recognition/DSS/
	
|-------------------------------------------------------------------------------------------------------------------------------------|

### Step 3: Install all necessary packages using the requirements.txt file.

	$ pip install -r requirements.txt
	
|-------------------------------------------------------------------------------------------------------------------------------------|
	
### Step 4: Run main.py to execute the end-to-end pipeline.

	$ python3 src/main.py --data_path 'DATA_PATH'
	
  where 'DATA_PATH' is the path to the image-data folder as string.

|-------------------------------------------------------------------------------------------------------------------------------------|

