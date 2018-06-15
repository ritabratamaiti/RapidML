# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:58:51 2018
@author: Ritabrata Maiti
"""
import pandas 
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import dill
from tpot import TPOTClassifier
from tpot import TPOTRegressor
import os
import sklearn
import numpy

d = defaultdict(LabelEncoder)
X = []
Y = []

print("\nRapidML, Version: 0.1, Author: Ritabrata Maiti")

def rapid_classifier(df, model = TPOTClassifier(generations=5, population_size=50, verbosity=2), name = "RapidML_Files"):
    print('\nUsing the RapidML Classifier; Experimental, For Issues Contact Author: ritabratamaiti@hiretrex.com')
    print("Label Encoding is being done....")
    if(type(model) != TPOTClassifier):
        raise ValueError('Error!! Model must be a TPOTClassifier')
    #Labelencoding the table
    df2 = df.values
    df_empty = df[0:0]
    fit = df.apply(lambda x: d[x.name].fit_transform(x))
    df = fit.values
    
    #getting X and Y, for training the classifier
    X = df[:, :(df.shape[1]-1)]
    Y = df[:, df.shape[1]-1]
    
    newpath = name 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    #pickling the dictionary d
    dill_file = open(name+"/d", "wb")
    dill_file.write(dill.dumps(d))
    dill_file.close()
    
    #pickling the skeletal dataframe df_empty
    dill_file = open(name+"/df", "wb")
    dill_file.write(dill.dumps(df_empty))
    dill_file.close()
    
    #training and pickling the classifier
    model.fit(X, Y)
    dill_file = open(name+"/model", "wb") 
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()
    
    dill_file = open(name+"/f", "wb")
    print("\nSample Output from input dataframe: ")
    print(','.join(str(e) for e in df2[2]))
    dill_file.write(dill.dumps(','.join(str(e) for e in df2[2])))
    dill_file.close()
    
    str1 = '''
#RD_AML created by Ritabrata Maiti
#Version: 1.0.0

from flask import Flask, request
import dill
import helper


app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
  return "RapidML, Project Version: 1.0.0"


@app.route('/query', methods=['GET', 'POST'])
def query_example():
    req = request.args['ip']
    dill_file = open("f", "rb")
    f = dill.load(dill_file)
    dill_file.close()
    l = [float(e) if e.isdigit() else e for e in f.split(',')]
    req = req + ',' + str(l[-1])
    dill_file = open("f", "wb")
    dill_file.write(dill.dumps(req))
    dill_file.close()
    helper.predictor()
    file = open('result.txt','r') 
    res = file.read()
    file.close()
    return res

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''
    
    file = open(name+"/API.py", "w")
    file.write(str1)
    file.close()
    
    str1 = '''
#RD_AML created by Ritabrata Maiti
#Version: 1.0.0

import dill
import pandas as pd  

def predictor():
 
    def fopen(str1, str2):
        dill_file = open(str1, str2)
        d = dill.load(dill_file)
        dill_file.close()
        return d
    
    d = fopen("d", "rb")
    df = fopen("df", "rb")
    f = fopen("f", "rb")
    model = fopen("model", "rb")
    l = [float(e) if e.isdigit() else e for e in f.split(',')]
    df.loc[0] = l
    fit = df.apply(lambda x: d[x.name].transform(x))
    df1 = fit.values
    X = df1[:, :(df1.shape[1]-1)]
    p = model.predict(X)
    p = d[list(df)[-1]].inverse_transform(p)
    file = open('result.txt','w') 
    file.write(str(p[0]))
    file.close()
    return 0

predictor()
    '''
    
    file = open(name+"/helper.py", "w")
    file.write(str1)
    file.close()
    
    return(model.fitted_pipeline_)


def rapid_regressor(df, le='No', model = TPOTRegressor(generations=5, population_size=50, verbosity=2), name="RapidML_Files"):
    
    print('\nUsing RapidML Regressor; Experimental, For Issues Contact Author: ritabratamaiti@hiretrex.com')
    
    if(type(model) != TPOTRegressor):
        raise ValueError('\nError!! Model must be a TPOTRegressor')
    df2 = df.values
    df_empty = df[0:0]   
        
    if(le == 'Yes'):        
        print("Label Encoding is being done....")
        #Labelencoding the table
        fit = df.apply(lambda x: d[x.name].fit_transform(x))
        df = fit.values
        #pickling the dictionary d
        dill_file = open(name+"/d", "wb")
        dill_file.write(dill.dumps(d))
        dill_file.close()

    if(le == 'No'):
        print('\nContinuing without label encoding')
        
    #getting X and Y, for training the classifier
    df = df.values
    X = df[:, :(df.shape[1]-1)]
    Y = df[:, df.shape[1]-1]
    
        
    newpath = name
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    
    #pickling the skeletal dataframe df_empty
    dill_file = open(name+"/df", "wb")
    dill_file.write(dill.dumps(df_empty))
    dill_file.close()
    
    #training and pickling the regressor
    model.fit(X, Y)
    dill_file = open(name+"/model", "wb") 
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()
    
    dill_file = open(name+"/f", "wb")
    print("\nSample Output from input dataframe: ")
    print(','.join(str(e) for e in df2[2]))
    dill_file.write(dill.dumps(','.join(str(e) for e in df2[2])))
    dill_file.close()
    
    str1 = '''
#RD_AML created by Ritabrata Maiti
#Version: 1.0.0

from flask import Flask, request
import dill
import helper


app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
  return "RapidML, Project Version: 1.0.0"


@app.route('/query', methods=['GET', 'POST'])
def query_example():
    req = request.args['ip']
    dill_file = open("f", "rb")
    f = dill.load(dill_file)
    dill_file.close()
    l = [float(e) if e.isdigit() else e for e in f.split(',')]
    req = req + ',' + str(l[-1])
    dill_file = open("f", "wb")
    dill_file.write(dill.dumps(req))
    dill_file.close()
    helper.predictor()
    file = open('result.txt','r') 
    res = file.read()
    file.close()
    return res

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    '''
    
    file = open(name+"/API.py", "w")
    file.write(str1)
    file.close()
    
    str1 = '''
#RD_AML created by Ritabrata Maiti
#Version: 1.0.0

import dill
import pandas as pd  
import os

def predictor():
 
    def fopen(str1, str2):
        dill_file = open(str1, str2)
        d = dill.load(dill_file)
        dill_file.close()
        return d

    df = fopen("df", "rb")
    f = fopen("f", "rb")
    model = fopen("model", "rb")        
    l = [float(e) if e.isdigit() else e for e in f.split(',')]
    df.loc[0] = l

    if(os.path.isfile('d')):
        d = fopen("d", "rb")          
        fit = df.apply(lambda x: d[x.name].transform(x))
    else:
        fit = df
        
    df1 = fit.values
    X = df1[:, :(df1.shape[1]-1)]
    p = model.predict(X)
    if(os.path.isfile('d')):
            p = d[list(df)[-1]].inverse_transform(p)
    file = open('result.txt','w') 
    file.write(str(p[0]))
    file.close()
    return 0

predictor()
    '''
    
    file = open(name+"/helper.py", "w")
    file.write(str1)
    file.close()
    
    return(model.fitted_pipeline_)


def rapid_regressor_arr(X, Y, model = TPOTRegressor(generations=5, population_size=50, verbosity=2), name="RapidML_Files"):
    
    print('\nUsing RapidML Regressor with arrays, Inputs will not be label encoded.; Experimental, For Issues Contact Author: ritabratamaiti@hiretrex.com')
    
    if(type(model) != TPOTRegressor):
        raise ValueError('\nError!! Model must be a TPOTRegressor')
         
    
    newpath = name
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    str1 = '''
from flask import Flask, request
import dill
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
  return "RapidML, Project Version: 1.0.0"


@app.route('/query', methods=['GET', 'POST'])
def query_example():
    req = request.args['ip']
    dill_file = open("model", "rb")
    model = dill.load(dill_file)
    dill_file.close()
    l = []
    for e in req.split(","):
           l.append(float(e))
           
    res = model.predict(np.reshape(l, (1,-1)))
    l = []
    return str(res[0])

if __name__ == '__main__':
    app.run(debug=True, port=8080)
         
    '''
    
    file = open(name+"/API.py", "w")
    file.write(str1)
    file.close()

    
    #training and pickling the regressor
    model.fit(X, Y)
    dill_file = open(name+"/model", "wb") 
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()
    
       
    return(model.fitted_pipeline_)


def rapid_classifier_arr(X, Y, model = TPOTRegressor(generations=5, population_size=50, verbosity=2), name="RapidML_Files"):
    
    print('\nUsing RapidML Classifier with arrays, Inputs will not be label encoded.; Experimental, For Issues Contact Author: ritabratamaiti@hiretrex.com')
    
    if(type(model) != TPOTClassifier):
        raise ValueError('\nError!! Model must be a TPOTClassifier')
         
    
    newpath = name
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    str1 = '''
from flask import Flask, request
import dill
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
  return "RapidML, Project Version: 1.0.0"


@app.route('/query', methods=['GET', 'POST'])
def query_example():
    req = request.args['ip']
    dill_file = open("model", "rb")
    model = dill.load(dill_file)
    dill_file.close()
    l = []
    for e in req.split(","):
           l.append(float(e))
           
    res = model.predict(np.reshape(l, (1,-1)))
    l = []
    return str(res[0])

if __name__ == '__main__':
    app.run(debug=True, port=8080)
       
    '''
    
    file = open(name+"/API.py", "w")
    file.write(str1)
    file.close()
  
    #training and pickling the classifier
    model.fit(X, Y)
    dill_file = open(name+"/model", "wb") 
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()
    
       
    return(model.fitted_pipeline_)
    
def rapid_udm(df, model, le='No', name="RapidML_Files"):
    
    print('\nRapidML User Defined Models, note that the model provided by the user should be a Scikit-Learn model or should work similarly to one.; Experimental, For Issues Contact Author: ritabratamaiti@hiretrex.com')
    
    df2 = df.values
    df_empty = df[0:0]   
        
    if(le == 'Yes'):        
        #Labelencoding the table
        print("Label Encoding is being done....")
        fit = df.apply(lambda x: d[x.name].fit_transform(x))
        df = fit.values
        #pickling the dictionary d
        dill_file = open(name+"/d", "wb")
        dill_file.write(dill.dumps(d))
        dill_file.close()

    if(le == 'No'):
        print('\nContinuing without label encoding')
        
    #getting X and Y, for training the classifier
    df = df.values
    X = df[:, :(df.shape[1]-1)]
    Y = df[:, df.shape[1]-1]
    
        
    newpath = name
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    
    #pickling the skeletal dataframe df_empty
    dill_file = open(name+"/df", "wb")
    dill_file.write(dill.dumps(df_empty))
    dill_file.close()
    
    #training and pickling the model
    model.fit(X, Y)
    dill_file = open(name+"/model", "wb") 
    dill_file.write(dill.dumps(model))
    dill_file.close()
    
    dill_file = open(name+"/f", "wb")
    print("\nSample Output from input dataframe: ")
    print(','.join(str(e) for e in df2[2]))
    dill_file.write(dill.dumps(','.join(str(e) for e in df2[2])))
    dill_file.close()
    
    str1 = '''
from flask import Flask, request
import dill
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
  return "RapidML, Project Version: 1.0.0"


@app.route('/query', methods=['GET', 'POST'])
def query_example():
    req = request.args['ip']
    dill_file = open("model", "rb")
    model = dill.load(dill_file)
    dill_file.close()
    l = []
    for e in req.split(","):
           l.append(float(e))
           
    res = model.predict(np.reshape(l, (1,-1)))
    l = []
    return str(res[0])

if __name__ == '__main__':
    app.run(debug=True, port=8080)
    '''
    
    file = open(name+"/API.py", "w")
    file.write(str1)
    file.close()
    
    str1 = '''
#RD_AML created by Ritabrata Maiti
#Version: 1.0.0

import dill
import pandas as pd  
import os

def predictor():
 
    def fopen(str1, str2):
        dill_file = open(str1, str2)
        d = dill.load(dill_file)
        dill_file.close()
        return d

    df = fopen("df", "rb")
    f = fopen("f", "rb")
    model = fopen("model", "rb")        
    l = [float(e) if e.isdigit() else e for e in f.split(',')]
    df.loc[0] = l

    if(os.path.isfile('d')):
        d = fopen("d", "rb")          
        fit = df.apply(lambda x: d[x.name].transform(x))
    else:
        fit = df
        
    df1 = fit.values
    X = df1[:, :(df1.shape[1]-1)]
    p = model.predict(X)
    if(os.path.isfile('d')):
            p = d[list(df)[-1]].inverse_transform(p)
    file = open('result.txt','w') 
    file.write(str(p[0]))
    file.close()
    return 0

predictor()
    '''
    
    file = open(name+"/helper.py", "w")
    file.write(str1)
    file.close()
    
    return(model)

def rapid_udm_arr(X, Y, model, name="RapidML_Files"):
    
    print('\nRapidML User Defined Models, Inputs will not be label encoded; note that the model provided by the user should be a Scikit_learn model and not a TPOT object.; Experimental, For Issues Contact Author: ritabratamaiti@hiretrex.com')
             
    newpath = name
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    str1 = '''
from flask import Flask, request
import dill
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
  return "RapidML, Project Version: 1.0.0"


@app.route('/query', methods=['GET', 'POST'])
def query_example():
    req = request.args['ip']
    dill_file = open("model", "rb")
    model = dill.load(dill_file)
    dill_file.close()
    l = []
    for e in req.split(","):
           l.append(float(e))
           
    res = model.predict(np.reshape(l, (1,-1)))
    l = []
    return str(res[0])

if __name__ == '__main__':
    app.run(debug=True, port=8080)
    '''
    
    file = open(name+"/API.py", "w")
    file.write(str1)
    file.close()
  
    #training and pickling the model
    model.fit(X, Y)
    dill_file = open(name+"/model", "wb") 
    dill_file.write(dill.dumps(model))
    dill_file.close()
    
    
    return(model)

