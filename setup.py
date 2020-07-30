from setuptools import setup, find_packages

setup(
    name='emotion_recognition',
    version='0.1.0',
    description="Tools for training and testing emotion recognition models and datasets.",
    author='Aaron Keesing',
    url='https://github.com/agkphysics/emotion',
    python_requires='>=3.6',
    packages=find_packages(include='emotion_recognition')
)
