# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 11:23:02 2018

@author: Ritabrata Maiti
"""

from setuptools import setup

setup(name='RapidML',
      version='1.0.2',
      description='RapidML is your Smart Machine Learning assistant that not only automates the creation of machine learning models but also enables you to easily deploy the models to the cloud. Find the documentation at: https://ritabratamaiti.github.io/RapidML',
      url='https://github.com/ritabratamaiti/RapidML',
      long_description = '''
What is RapidML?

Well, RapidML is your Smart Machine Learning assistant that not only automates the creation of machine learning models but also enables you to easily deploy the models to the cloud.

RapidML is perfect for Python programmers at all levels, ranging from beginners who want to get into Data Science and Machine Learning to intermediate and advanced programmers who want to bring Machine Learning to consumer and industry usage applications.

Apart from making predictions in Python, RapidML models can be exported as Web APIs to develop Machine Learning applications in a wide variety of platforms, such as Javascript, Android, iOS…. and almost everything else which can make and receive web requests!

Documentation: https://ritabratamaiti.github.io/RapidML

Email: ritabratamaiti@hiretrex.com

Issues Page: https://github.com/ritabratamaiti/RapidML/issues

Twitter: https://twitter.com/ritabratamaiti
      ''',
      author='Ritabrata Maiti',
      author_email='ritabratamaiti@hiretrex.com',
      license='LGPLv3',
      packages=['RapidML'],
      zip_safe=False,
    install_requires=['numpy>=1.12.1',
                    'scipy>=0.19.0',
                    'scikit-learn>=0.18.1',
                    'deap>=1.0',
                    'update_checker>=0.16',
                    'tqdm>=4.11.2',
                    'stopit>=1.1.1',
                    'pandas>=0.20.2',
                    'TPOT>=0.9',
					'dill>=0.2.8'],
        extras_require={
        'xgboost': ['xgboost==0.6a2'],
        'skrebate': ['skrebate>=0.3.4'],
        'mdr': ['scikit-mdr>=0.4.4']
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=['Machine Learning API','pipeline optimization', 'hyperparameter optimization', 'data science', 'machine learning', 'genetic programming', 'evolutionary computation'],
)
