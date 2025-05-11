from setuptools import setup, find_packages

setup(
    name='cleanGPT',
    version='0.1.0',
    packages=find_packages(include=['mytokenizers', 'utils', 'inference', 'model', 'trainers']),
    install_requires=[
        'torch>=1.10',
        'datasets>=2.0.0',
        'transformers>=4.20.0',
        'tqdm>=4.60.0',
        'matplotlib'
    ],
)
