# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:58:51 2018

@author: Ritabrata Maiti
"""
import pandas as pd   
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import dill
from tpot import TPOTClassifier
from tpot import TPOTRegressor
import os

d = defaultdict(LabelEncoder)
X = []
Y = []

def rapid_classifier(df, model = TPOTClassifier(generations=5, population_size=50, verbosity=2)):
    #Labelencoding the table
    df2 = df.values
    df_empty = df[0:0]
    fit = df.apply(lambda x: d[x.name].fit_transform(x))
    df = fit.values
    
    #getting X and Y, for training the classifier
    X = df[:, :(df.shape[1]-1)]
    Y = df[:, df.shape[1]-1]
    
    newpath = r'webml' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    #pickling the dictionary d
    dill_file = open("webml/d", "wb")
    dill_file.write(dill.dumps(d))
    dill_file.close()
    
    #pickling the skeletal dataframe df_empty
    dill_file = open("webml/df", "wb")
    dill_file.write(dill.dumps(df_empty))
    dill_file.close()
    
    #training and pickling the classifier
    model.fit(X, Y)
    dill_file = open("webml/model", "wb") 
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()
    
    dill_file = open("webml/f", "wb")
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
      return "RD_AML, Version: 1.0.0"
    
    
    @app.route('/query', methods=['GET', 'POST'])
    def query_example():
        req = request.args['ip']
        dill_file = open("f", "rb")
        f = dill.load(dill_file)
        dill_file.close()
        l = [int(e) if e.isdigit() else e for e in f.split(',')]
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
    
    file = open("webml/API.py", "w")
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
        l = [int(e) if e.isdigit() else e for e in f.split(',')]
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
    
    file = open("webml/helper.py", "w")
    file.write(str1)
    file.close()


def rapid_regressor(df, model = TPOTRegressor(generations=5, population_size=50, verbosity=2)):
    df2 = df.values
    df_empty = df[0:0]    
    #Labelencoding the table
    
    fit = df.apply(lambda x: d[x.name].fit_transform(x))
    df = fit.values
    
    #getting X and Y, for training the classifier
    X = df[:, :(df.shape[1]-1)]
    Y = df[:, df.shape[1]-1]
    
    
    newpath = r'webml' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    #pickling the dictionary d
    dill_file = open("webml/d", "wb")
    dill_file.write(dill.dumps(d))
    dill_file.close()
    
    #pickling the skeletal dataframe df_empty
    dill_file = open("webml/df", "wb")
    dill_file.write(dill.dumps(df_empty))
    dill_file.close()
    
    #training and pickling the classifier
    model.fit(X, Y)
    dill_file = open("webml/model", "wb") 
    dill_file.write(dill.dumps(model.fitted_pipeline_))
    dill_file.close()
    
    dill_file = open("webml/f", "wb")
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
      return "RD_AML, Version: 1.0.0"
    
    
    @app.route('/query', methods=['GET', 'POST'])
    def query_example():
        req = request.args['ip']
        dill_file = open("f", "rb")
        f = dill.load(dill_file)
        dill_file.close()
        l = [int(e) if e.isdigit() else e for e in f.split(',')]
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
    
    file = open("webml/API.py", "w")
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
        l = [int(e) if e.isdigit() else e for e in f.split(',')]
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
    
    file = open("webml/helper.py", "w")
    file.write(str1)
    file.close()
    
