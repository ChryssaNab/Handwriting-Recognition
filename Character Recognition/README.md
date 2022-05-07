# Character Recognition for Dead Sea Scrolls

### [**Contents**](#) <a name="cont"></a>
1. [Project Description](#descr)
2. [Usage](#Run)
3. [Benchmarking](#Ben)
4. [Team](#Team)
5. [See also](#ext) 

### [Usage](#) <a name="Run"></a>

**1.** We assume that Python3 is already installed on the device and the Dead Sea Scrolls (DDS) dataset is downloaded.

**2.** Clone this repository: 
``` shell
$ git clone https://github.com/ChryssaNab/Handwriting_Recognition-RUG.git
$ cd Handwriting_Recognition-RUG/Character\ Recognition/
```

**3.** Install all necessary packages using the **requirements.txt** file:
``` shell
$ pip install -r requirements.txt
```

**4.** Change the absolute paths of DDS input data and results using the **/src/opts.py**.

**5.** Run **main.py** to segment DDS test images into lines and words.

``` shell
 $ python3 src/main.py
 ```

