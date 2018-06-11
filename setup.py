# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 11:23:02 2018

@author: Ritabrata Maiti
"""

from setuptools import setup

setup(name='RapidML',
      version='0.1',
      description='Automated Machine Learning Model Creation, with corresponding model access via cloud through API',
      url='https://github.com/ritabratamaiti/RapidML',
      long_description = '''
      This is a python tool that determines the best machine learing model using genetic programming, and then subsequently produces an API and saves models for further use.
      Contact me for further queries.
      Email: ritabratamaiti@hiretrex.com
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
                    'TPOT>=0.9'],
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
