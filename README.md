# Easy_Facial_Recognition

### Motivation

OpenCV and Dlib offer some of the best facial recognition algorithms for free, as of this writing.  Unfortunately, I've ended up having to write a lot of duplicate code for the most common facial recognition and image manipulation operations across my various programs.  So I've built wrapper classes and functions into this library to make the process easier.

### Installation

As of now, Easy Facial Recognition has only been tested to run on Windows 10 with Anaconda3 under Python 3.6.  I may test other versions of Python and other OS's in the future and report my results here.  You're also welcome to try it out on other setups - feel free to report any issues here.

##### Windows and Mac:

First, download and install [Anaconda3](https://www.anaconda.com/download/)

Open a Terminal or Command Prompt

You'll need to set up a few dependencies before running:

`conda install opencv path.py numpy`

`conda install -c conda-forge dlib Pillow`

`pip install piexif`

Then download and install Easy Facial Recognition:

`git clone https://github.com/xjdeng/Easy_Facial_Recognition`

`cd Easy_Facial_Recognition`

`pip install .`

### What can you do with this module?

First, open a terminal or command prompt in a working directory of your choice.  Then run the following to copy some test images to this directory.  **Do this before running any of the following examples!**  Then run the following sections **in order.**

`from Easy_Facial_Recognition import test`

`import Easy_Facial_Recognition`

`test.copytests()`

##### Initialize a Face with the EasyImage object:

`import cv2`

`angryimg = Easy_Facial_Recognition.EasyImage(cv2.imread("tests/angry-2191104_640.jpg"))`

##### Initialize an image file using the EasyImageFile object:

`woman = Easy_Facial_Recognition.EasyImageFile("tests/woman-3046960_640.jpg")`

##### Detect Faces (EasyImage and EasyImageFile objects):

`angryface = angryimg.detect_faces()`

`import time`

`t0 = time.time()`

`womanface = woman.detect_faces()`

`t1 = time.time()`

`print(str(t1 - t0) + " seconds to detect face.")`

##### Redetect faces in an EasyImageFile object again:

`t0 = time.time()`

`womanface2 = woman.detect_faces()`

`t1 = time.time()`

`print(str(t1 - t0) + " seconds to detect face this time")`

Since the faces were stored using EXIF tags in the file tests/woman-3046960_640.jpg, it took less time after running detect_faces() again.

##### Draw boxes around faces:

`angryfacebox = angryimg.draw_faces()`

`womanbox = woman.draw_faces()`

`cv2.imshow("face 1", angryfacebox.getimg())`

`cv2.waitKey(0)`





