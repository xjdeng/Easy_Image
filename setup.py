from setuptools import setup

setup(name="Easy_Facial_Recognition",version='0.1',description='Wrapper\
      classes to make facial recognition with OpenCV and Dlib easier',\
      author='Xiaojian Deng',author_email="xjd001@gmail.com",license='MIT',\
      packages=['Easy_Facial_Recognition'],include_package_data=True,\
      package_data={'Easy_Facial_Recognition': ['tests/*','haarcascades/*',\
                                                'lbpcascades/*']},
      url="https://github.com/xjdeng/Easy_Facial_Recognition",zip_safe=False)