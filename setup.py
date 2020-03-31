from setuptools import setup

setup(
    name='emotion_recognition',
    version='0.0.0',
    description="Tools for creating emotion recognisers from speech.",
    author='Aaron Keesing',
    url='https://github.com/agkphysics/emotion',
    python_requires='>=3.6',
    package_dir={'': 'scripts'},
    packages=['emotion_recognition', 'emotion_recognition.tensorflow']
)
