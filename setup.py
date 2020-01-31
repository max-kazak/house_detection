from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project detects houses on the satellite images. It uses neural network trained on _Inria Aerial Image Labeling Dataset_.',
    author='Maxim Kazakov',
    license='MIT',
)
