*** README - IAM/Task3 ***

Required packages can be installed with:
>> pip install -r requirements.txt

For help:
>> python main.py -h

Prints this message:
    usage: main.py [-h] [--mode {train,test}] [--debug] [--final] [path]

    args for IAM main

    positional arguments:
      path                 path to 'img' (test) or 'IAM-data' (train)

    optional arguments:
      -h, --help           show this help message and exit
      --mode {train,test}  train or test model (default test)
      --debug              use debug mode (default False)
      --final              train model on the whole data set (default False)


For testing:
>> python main.py [path/to/iam/data/img]

Test output:
A folder 'results' in the same directory as the source files which contains
a txt file for each image with the corresponding prediction.
