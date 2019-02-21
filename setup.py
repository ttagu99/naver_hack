#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch1.0keras2.2
from distutils.core import setup

setup(
    name='nsml vision hackathon',
    version='1.0',
    description='nsml vision hackathon',
    install_requires=[
        'scikit-learn',
        'imgaug==0.2.7',
        'image-classifiers',
        'pandas',
        'kito',
        'psutil'
	    ]
)
