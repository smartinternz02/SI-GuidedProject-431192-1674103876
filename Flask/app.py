import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn.externals as extjoblib
import joblib
app = Flask(__name__)
mode = joblib.load('Random.pkl')
model1 =joblib.load('Linear.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''


    int_features = [[float(x) for x in request.form.values()]]
    
  
    p = mode.predict(int_features)[0]
    prediction1 = model1.predict(int_features)[0]

  
    return render_template('index.html', prediction_text='Random Forest: GDP of the country is  $ {}'.format(p), prediction_text2='Linear Regression: GDP of the country is  $ {}'.format(prediction1))

if __name__ == "__main__":
    app.run(debug=True,port=8000)