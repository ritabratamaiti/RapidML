# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:58:51 2018

Eagle included for good luck ^_^

                               /T /I
                              / |/ | .-~/
                          T\ Y  I  |/  /  _
         /T               | \I  |  I  Y.-~/
        I l   /I       T\ |  |  l  |  T  /
     T\ |  \ Y l  /T   | \I  l   \ `  l Y
 __  | \l   \l  \I l __l  l   \   `  _. |
 \ ~-l  `\   `\  \  \ ~\  \   `. .-~   |
  \   ~-. "-.  `  \  ^._ ^. "-.  /  \   |
.--~-._  ~-  `  _  ~-_.-"-." ._ /._ ." ./
 >--.  ~-.   ._  ~>-"    "\   7   7   ]
^.___~"--._    ~-{  .-~ .  `\ Y . /    |
 <__ ~"-.  ~       /_/   \   \I  Y   : |
   ^-.__           ~(_/   \   >._:   | l______
       ^--.,___.-~"  /_/   !  `-.~"--l_ /     ~"-.
              (_/ .  ~(   /'     "~"--,Y   -=b-. _)
               (_/ .  \  :           / l      c"~o \
                \ /    `.    .     .^   \_.-~"~--.  )
                 (_/ .   `  /     /       !       )/
                  / / _.   '.   .':      /        '
                  ~(_/ .   /    _  `  .-<_
                    /_/ . ' .-~" `.  / \  \          ,z=.
                    ~( /   '  :   | K   "-.~-.______//
                      "-,.    l   I/ \_    __{--->._(==.
                       //(     \  <    ~"~"     //
                      /' /\     \  \     ,v=.  ((
                    .^. / /\     "  }__ //===-  `
                   / / ' '  "-.,__ {---(==-
                 .^ '       :  T  ~"   ll       -Row
                / .  .  . : | :!        \
               (_/  /   | | j-"          ~^



@author: Ritabrata Maiti
"""
import pandas
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import dill
import tpot
from tpot import TPOTClassifier
from tpot import TPOTRegressor
import os
import sklearn
import numpy

X = []
Y = []


class rml:    

    def put(self, mdl, d=None, s=1):
        if (s==1):
            self.model = mdl.fitted_pipeline_
            self.m_tpot = mdl
        else:
            self.model = mdl
            self.m_tpot = None
        self.d = d

    def le(self, df):
        if (self.d != None):
            d = self.d
            fit = df.apply(lambda x: d[x.name].transform(x))
            return (fit)
        else:
            print("\nModel wasn't Label Encoded....")


print("\nRapidML, Version: 1.0.2, Author: Ritabrata Maiti\n")
logo = '''
       .---.        .-----------
      /     \  __  /    ------
     / /     \(  )/    -----
    //////   ' \/ `   ---
   //// / // :    : ---
  // /   /  /`    '--
 //          //..\\
        ====UU====UU====
            '//||\\\`
              ''``
'''
print(logo)


def rapid_classifier(df,
                     le='Yes',
                     model=TPOTClassifier(
                         generations=5, population_size=50, verbosity=2),
                     name="RapidML_Files"):
    print(
        '\nUsing the RapidML Classifier; Experimental, For Issues Visit: https://github.com/ritabratamaiti/RapidML/issues or Contact Author: ritabratamaiti@hiretrex.com'
    )
    d = defaultdict(LabelEncoder)
    if (type(model) != TPOTClassifier):
        raise ValueError('Error!! Model must be a TPOTClassifier')

    df2 = df
    df_empty = df[0:0]
    os.makedirs(name, exist_ok=True)
    if (le == 'Yes'):
        print("Label Encoding is being done....")
        #Labelencoding the table
        fit = df.apply(lambda x: d[x.name].fit_transform(x))

        df = fit.values
        #pickling the dictionary d
        dill_file = open(name + "/d", "wb")
        dill_file.write(dill.dumps(d))
        dill_file.close()

    elif (le == 'No'):
        print('\nContinuing without label encoding')
        df = df.values
    else:
        raise ValueError('Unexpected value for labelencoding options!!')
    #getting X and Y, for training the classifier

    X = df[:, :(df.shape[1] - 1)]
    Y = df[:, df.shape[1] - 1]

    #pickling the skeletal dataframe df_empty
    dill_file = open(name + "/df", "wb")
    dill_file.write(dill.dumps(df_empty))
    dill_file.close()

    #training and pickling the classifier
    print('\nTraining....\n')
    model.fit(X, Y)
    dill_file = open(name + "/model", "wb")
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()

    dill_file = open(name + "/f", "wb")
    print("\nSample Output from input dataframe: ")
    print(','.join(str(e) for e in df2.values[2]))
    dill_file.write(dill.dumps(','.join(str(e) for e in df2.values[2])))
    dill_file.close()

    dill_file = open(name + "/dt", "wb")
    dt = []

    df2 = df2.astype('object')

    for col in df2:
        dt.append(type(df2.loc[0, col]))


    dill_file.write(dill.dumps(dt))
    dill_file.close()

    str1 = '''
#RD_AML created by Ritabrata Maiti
#Version: 1.0.2

from flask import Flask, request
import dill
import helper
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
  return "RapidMLex, Project Version: 1.0.2"


@app.route('/query', methods=['GET', 'POST'])
def query_example():
    req = request.args['ip']
    dill_file = open("f", "rb")
    f = dill.load(dill_file)
    dill_file.close()
    dill_file = open("dt", "rb")
    dt = dill.load(dill_file)
    dill_file.close()
    l = []
    i=0
    for e in f.split(','):
           k = dt[i]
           l.append(k(e))
           i+=1
    req = req + ',' + l[-1]
    dill_file = open("f", "wb")
    dill_file.write(dill.dumps(req))
    dill_file.close()
    helper.predictor()
    file = open('result.txt','r') 
    res = file.read()
    file.close()
    return res


if __name__ == '__main__':
    app.run(debug=True, port=8080)
'''

    file = open(name + "/API.py", "w")
    file.write(str1)
    file.close()

    str1 = '''
#RapidML created by Ritabrata Maiti
#Version: 1.0.2

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
    dt = fopen("dt", "rb")   
    l = []
    i = 0
    for e in dt:
           
        l.append(e(f.split(',')[i]))     
        i+=1


    if(os.path.isfile('d')):
        d = fopen("d", "rb")      
        df.loc[0] = l
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


    '''

    file = open(name + "/helper.py", "w")
    file.write(str1)
    file.close()

    ob = rml()

    if (le == 'Yes'):
        ob.put(model, d)
    else:
        ob.put(model)
    return (ob)


def rapid_regressor(df,
                    le='No',
                    model=TPOTRegressor(
                        generations=5, population_size=50, verbosity=2),
                    name="RapidML_Files"):

    print(
        '\nUsing RapidML Regressor; Experimental, For Issues Visit: https://github.com/ritabratamaiti/RapidML/issues or Contact Author: ritabratamaiti@hiretrex.com'
    )
    d = defaultdict(LabelEncoder)
    if (type(model) != TPOTRegressor):
        raise ValueError('\nError!! Model must be a TPOTRegressor')
    df2 = df
    df_empty = df[0:0]
    os.makedirs(name, exist_ok=True)
    if (le == 'Yes'):
        print("Label Encoding is being done....")
        #Labelencoding the table
        fit = df.apply(lambda x: d[x.name].fit_transform(x))

        df = fit.values
        #pickling the dictionary d
        dill_file = open(name + "/d", "wb")
        dill_file.write(dill.dumps(d))
        dill_file.close()

    elif (le == 'No'):
        print('\nContinuing without label encoding')
        df = df.values
    else:
        raise ValueError('Unexpected value for labelencoding options!!')
    #getting X and Y, for training the classifier

    X = df[:, :(df.shape[1] - 1)]
    Y = df[:, df.shape[1] - 1]

    #pickling the skeletal dataframe df_empty
    dill_file = open(name + "/df", "wb")
    dill_file.write(dill.dumps(df_empty))
    dill_file.close()

    #training and pickling the classifier
    print('\nTraining....\n')
    model.fit(X, Y)
    dill_file = open(name + "/model", "wb")
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()

    dill_file = open(name + "/f", "wb")
    print("\nSample Output from input dataframe: ")
    print(','.join(str(e) for e in df2.values[2]))
    dill_file.write(dill.dumps(','.join(str(e) for e in df2.values[2])))
    dill_file.close()

    dill_file = open(name + "/dt", "wb")
    dt = []

    df2 = df2.astype('object')

    for col in df2:
        dt.append(type(df2.loc[0, col]))

    
    dill_file.write(dill.dumps(dt))
    dill_file.close()

    str1 = '''
#RD_AML created by Ritabrata Maiti
#Version: 1.0.2

from flask import Flask, request
import dill
import helper
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
  return "RapidMLex, Project Version: 1.0.2"


@app.route('/query', methods=['GET', 'POST'])
def query_example():
    req = request.args['ip']
    dill_file = open("f", "rb")
    f = dill.load(dill_file)
    dill_file.close()
    dill_file = open("dt", "rb")
    dt = dill.load(dill_file)
    dill_file.close()
    l = []
    i=0
    for e in f.split(','):
           k = dt[i]
           l.append(k(e))
           i+=1
    req = req + ',' + l[-1]
    dill_file = open("f", "wb")
    dill_file.write(dill.dumps(req))
    dill_file.close()
    helper.predictor()
    file = open('result.txt','r') 
    res = file.read()
    file.close()
    return res


if __name__ == '__main__':
    app.run(debug=True, port=8080)
'''

    file = open(name + "/API.py", "w")
    file.write(str1)
    file.close()

    str1 = '''
#RapidML created by Ritabrata Maiti
#Version: 1.0.2

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
    dt = fopen("dt", "rb")   
    l = []
    i = 0
    for e in dt:
           
        l.append(e(f.split(',')[i]))     
        i+=1


    if(os.path.isfile('d')):
        d = fopen("d", "rb")      
        df.loc[0] = l
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


    '''

    file = open(name + "/helper.py", "w")
    file.write(str1)
    file.close()

    ob = rml()
    if (le == 'Yes'):
        ob.put(model, d)
    else:
        ob.put(model)
    return (ob)


def rapid_regressor_arr(X,
                        Y,
                        model=TPOTRegressor(
                            generations=5, population_size=50, verbosity=2),
                        name="RapidML_Files"):

    print(
        '\nUsing RapidML Regressor with arrays, Inputs will not be label encoded; Experimental, For Issues Visit: https://github.com/ritabratamaiti/RapidML/issues or Contact Author: ritabratamaiti@hiretrex.com'
    )

    if (type(model) != TPOTRegressor):
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
  return "RapidML, Project Version: 1.0.2"


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

    file = open(name + "/API.py", "w")
    file.write(str1)
    file.close()

    #training and pickling the regressor
    print('\nTraining....\n')
    model.fit(X, Y)
    dill_file = open(name + "/model", "wb")
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()

    ob = rml()
    ob.put(model)
    return (ob)


def rapid_classifier_arr(X,
                         Y,
                         model=TPOTClassifier(
                             generations=5, population_size=50, verbosity=2),
                         name="RapidML_Files"):

    print(
        '\nUsing RapidML Classifier with arrays, Inputs will not be label encoded; Experimental, For Issues Visit: https://github.com/ritabratamaiti/RapidML/issues or Contact Author: ritabratamaiti@hiretrex.com'
    )

    if (type(model) != TPOTClassifier):
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
  return "RapidML, Project Version: 1.0.2"


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

    file = open(name + "/API.py", "w")
    file.write(str1)
    file.close()

    #training and pickling the classifier
    print('\nTraining....\n')
    model.fit(X, Y)
    dill_file = open(name + "/model", "wb")
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()

    ob = rml()
    ob.put(model)
    return (ob)


def rapid_udm(df, model, le='No', name="RapidML_Files"):

    print(
        '\nUsing RapidML with an User Defined Model; Experimental, For Issues Visit: https://github.com/ritabratamaiti/RapidML/issues or Contact Author: ritabratamaiti@hiretrex.com'
    )
    d = defaultdict(LabelEncoder)

    df2 = df
    df_empty = df[0:0]
    os.makedirs(name, exist_ok=True)
    if (le == 'Yes'):
        print("Label Encoding is being done....")
        #Labelencoding the table
        fit = df.apply(lambda x: d[x.name].fit_transform(x))

        df = fit.values
        #pickling the dictionary d
        dill_file = open(name + "/d", "wb")
        dill_file.write(dill.dumps(d))
        dill_file.close()

    elif (le == 'No'):
        print('\nContinuing without label encoding')
        df = df.values()
    else:
        raise ValueError('Unexpected value for labelencoding options!!')
    #getting X and Y, for training the classifier

    X = df[:, :(df.shape[1] - 1)]
    Y = df[:, df.shape[1] - 1]

    #pickling the skeletal dataframe df_empty
    dill_file = open(name + "/df", "wb")
    dill_file.write(dill.dumps(df_empty))
    dill_file.close()

    #training and pickling the classifier
    print('\nTraining....\n')
    model.fit(X, Y)
    dill_file = open(name + "/model", "wb")
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()

    dill_file = open(name + "/f", "wb")
    print("\nSample Output from input dataframe: ")
    print(','.join(str(e) for e in df2.values[2]))
    dill_file.write(dill.dumps(','.join(str(e) for e in df2.values[2])))
    dill_file.close()

    dill_file = open(name + "/dt", "wb")
    dt = []

    df2 = df2.astype('object')

    for col in df2:
        dt.append(type(df2.loc[0, col]))

    
    dill_file.write(dill.dumps(dt))
    dill_file.close()

    str1 = '''
#RD_AML created by Ritabrata Maiti
#Version: 1.0.2

from flask import Flask, request
import dill
import helper
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
  return "RapidMLex, Project Version: 1.0.2"


@app.route('/query', methods=['GET', 'POST'])
def query_example():
    req = request.args['ip']
    dill_file = open("f", "rb")
    f = dill.load(dill_file)
    dill_file.close()
    dill_file = open("dt", "rb")
    dt = dill.load(dill_file)
    dill_file.close()
    l = []
    i=0
    for e in f.split(','):
           k = dt[i]
           l.append(k(e))
           i+=1
    req = req + ',' + l[-1]
    dill_file = open("f", "wb")
    dill_file.write(dill.dumps(req))
    dill_file.close()
    helper.predictor()
    file = open('result.txt','r') 
    res = file.read()
    file.close()
    return res


if __name__ == '__main__':
    app.run(debug=True, port=8080)
'''

    file = open(name + "/API.py", "w")
    file.write(str1)
    file.close()

    str1 = '''
#RapidML created by Ritabrata Maiti
#Version: 1.0.2

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
    dt = fopen("dt", "rb")   
    l = []
    i = 0
    for e in dt:
           
        l.append(e(f.split(',')[i]))     
        i+=1


    if(os.path.isfile('d')):
        d = fopen("d", "rb")      
        df.loc[0] = l
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


    '''

    file = open(name + "/helper.py", "w")
    file.write(str1)
    file.close()

    ob = rml()
    if (le == 'Yes'):
        ob.put(model, d, s=0)
    else:
        ob.put(model,s=0)
    return (ob)


def rapid_udm_arr(X, Y, model, name="RapidML_Files"):

    print(
        '\nUsing RapidML with User Defined Models and Arrays, Inputs will not be label encoded; note that the model provided by the user should be a Scikit_learn model and not a TPOT object; Experimental, For Issues Visit: https://github.com/ritabratamaiti/RapidML/issues or Contact Author: ritabratamaiti@hiretrex.com'
    )

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
  return "RapidML, Project Version: 1.0.2"


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

    file = open(name + "/API.py", "w")
    file.write(str1)
    file.close()

    #training and pickling the model
    print('\nTraining....\n')
    model.fit(X, Y)
    dill_file = open(name + "/model", "wb")
    dill_file.write(dill.dumps(model))
    dill_file.close()

    ob = rml()
    ob.put(model,s=0)
    return (ob)
