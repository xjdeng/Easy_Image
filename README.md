# Easy_Facial_Recognition

### Motivation

OpenCV and Dlib offer some of the best facial recognition algorithms for free, as of this writing.  Unfortunately, I've ended up having to write a lot of duplicate code for the most common facial recognition and image manipulation operations across my various programs.  So I've built wrapper classes and functions into this library to make the process easier.

### Installation

As of now, Easy Facial Recognition has only been tested to run on Windows 10 with Anaconda3 under Python 3.6.  I may test other versions of Python and other OS's in the future and report my results here.  You're also welcome to try it out on other setups - feel free to report any issues here.

You'll need to set up a few dependencies before running:

`conda install opencv path.py numpy`

`conda install -c conda-forge dlib Pillow`

`pip install piexif`

### What can you do with this module?

##### Initialize a Face with the EasyImage object:

##### Initialize a Face from an image file using the EasyImageFile object:

##### Detect Faces (EasyImage and EasyImageFile objects):

##### Redetect faces in an EasyImageFile object again:

##### Draw boxes around faces: