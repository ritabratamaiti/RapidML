=============
Using RapidML
=============

****************
Why use RapidML?
****************



Extremely easy to use and intuitive API
=======================================

RapidML can accept DataFrame inputs, which are essentially the easiest ways to programmatically represent .csv (Comma separated values) or .xlsx (Excel) files in Python. Since .csv files and .xlsx are the most commonly used file types in organizing and storing large amounts of data, popular python libraries like Pandas provide methods for converting these files into DataFrames in Python, leading to the popularity of the DataFrame datatype in Data Science and Machine Learning.

Not all data is numeric, and it is common for some categorical data to be in a textual format. However, many machine learning algorithms can only work with numeric data. RapidML easily solves this problem for data scientists by performing automatic label encoding on such data. An example of this could be a gender attribute (field) containing 'female', 'male' and 'other' as possible values, then RapidML can encode these as, say, {'female': 0, 'other': 1, 'male': 2}


Rapid development of Machine Learning Models and Web APIs
=========================================================


RapidML utilizes TPOT (Tree-based Pipeline Optimization Tool) as its backend for automated algorithm selection. TPOT utilizes genetic programming to select the best Scikit-Learn model and performs hyperparameter optimization on the model. RapidML then trains this model and uses it to perform future predictions.

Using the model, RapidML programmatically generates a Web API using the Flask framework, which can be easily self-hosted or uploaded to a remote server in order to make machine learning predictions in the cloud. Requests are quite easy to make, by using URL parameters in the ``/query?ip=`` part of the request. See making requests for more info.


Machine Learning models can be quickly deployed to production
=============================================================

This is an extension of the previous point and serves to highlight the fact that RapidML models can be used for easy and quick prototyping of production-level applications, on multiple platforms such as through the web, as well as Android and iOS applications, just to name a few.


RapidML is very versatile and it can be extended for use by more experienced Developers and Data Scientists
===========================================================================================================


If you prefer to select machine learning algorithms on your own, then RapidML allows you to train the model and run it in the cloud. For example, if you don't wish to use TPOT's automated machine learning, but would rather implement a neural network (Scikit-learn's implementation), then you can easily do so using RapidML's ``udm`` (user-defined model) function in order to run the neural network in the cloud.
RapidML's versatility makes it a tool that encompasses the differences between programmers, Data Scientists, and Web Developers at all expertise levels and enables everyone to take part in the exciting field of Machine Learning and Artificial Intelligence.


RapidML in Action
=================

RapidML has a wide variety of application, ranging from the field of medicine to scientific predictions in statistics, like a prediction of radiation emissions or meteorological prediction (weather prediction) and even predicting stocks prices. 
RapidML is already being used in the development of a Machine Learning Web API  that is utilized in an android application for ASD detection in adult patients. 

Find this project here: https://github.com/ritabratamaiti/Autism-Detection-API. `This project was developed to showcase RapidML's use cases, and shouldn't be use for making diagnosis without clinical trials, and permission of the author`.

*****************
RapidML with code
*****************

The following code should give a fair idea on RapidML usage. Note: Visit :ref:`Example` for more code samples and projects.

.. code-block:: python

       import RapidML
       import os
       import pandas as pd
       
       
       # This Autism Screening Adult Data Set is from UCI Machine Learning Repository and is available here: https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult
       
       df = pd.read_csv('out.csv')
       df = df.drop(columns = ['Unnamed: 0'])
       df.head()
       
       ml_model = RapidML.rapid_classifier(df,name='ASDapi')

*Note: The training data is an Autism Screening Adult DataSet from UCI Machine Learning Repository and is available here:* https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult

The code generates the following output. Here ``ml_model`` is assigned an ``rml`` object by the ``RapidML.rapid_classifier`` function. To learn about the ``rml`` class, as well as RapidML functions, go to :ref:`RapidML_API`.
    

.. code-block:: text

	    
    RapidML, Version: 0.1, Author: Ritabrata Maiti
    
    
           .---.        .-----------
          /     \  __  /    ------
         / /     \(  )/    -----
        //////   ' \/ `   ---
       //// / // :    : ---
      // /   /  /`    '--
     //          //..\
            ====UU====UU====
                '//||\\`
                  ''``
    
    Warning: xgboost.XGBClassifier is not available and will not be used by TPOT.
    Warning: xgboost.XGBRegressor is not available and will not be used by TPOT.
    Warning: xgboost.XGBRegressor is not available and will not be used by TPOT.
    Warning: xgboost.XGBRegressor is not available and will not be used by TPOT.
    
    Using the RapidML Classifier; Experimental, For Issues Contact Author: ritabratamaiti@hiretrex.com
    Label Encoding is being done....
    
    Training....
    
    Generation 1 - Current best internal CV score: 1.0                            
    Generation 2 - Current best internal CV score: 1.0                            
    Generation 3 - Current best internal CV score: 1.0                            
    Generation 4 - Current best internal CV score: 1.0                            
    Generation 5 - Current best internal CV score: 1.0                            
                                                                                  
    Best pipeline: DecisionTreeClassifier(input_matrix, criterion=entropy, max_depth=2, min_samples_leaf=4, min_samples_split=6)
    
    Sample Output from input dataframe: 
    1,1,0,1,0,0,1,1,0,1,6,35.0,f,White-European,no,yes,United States,no,Self,NO
