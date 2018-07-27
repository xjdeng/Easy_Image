from setuptools import setup
#https://docs.python.org/2/distutils/setupscript.html
#Info on including data files

setup(name="Easy_Image",version='0.1',description='Wrapper\
      classes to make facial recognition with OpenCV and Dlib easier and \
      to automatically detect and classify images.',\
      author='Xiaojian Deng',author_email="xjd001@gmail.com",license='MIT',\
      packages=['Easy_Image'],include_package_data=True,\
      package_data={'Easy_Image': ['tests/*','haarcascades/*',\
                                   'misc/*',    'lbpcascades/*']},
      url="https://github.com/xjdeng/Easy_Image",zip_safe=False)