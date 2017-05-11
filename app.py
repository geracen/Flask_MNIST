from flask import Flask, request,jsonify
import os
import pickle
import numpy as np

with open('mymodel.pkl','rb') as f:
    model = pickle.load(f)

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    return "Hello Flask"

@app.route('/predict',methods=['GET','POST'])
def predict():
    params=request.json['input']
    y = model.predict(params)[0]
    return jsonify({"prediction":np.int(y)})

if __name__=='__main__':
    app.run(debug=True)


