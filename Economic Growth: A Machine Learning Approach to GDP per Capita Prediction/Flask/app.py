import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn.externals as extjoblib
import joblib
app = Flask(__name__)
model = joblib.load('Random.pkl')
model1 = joblib.load('Linear.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
  
    prediction = model.predict(final_features)
    prediction1 = model1.predict(final_features)

    output = round(prediction[0], 2)
    output1 = round(prediction1[0], 2)
    return render_template('index.html', prediction_text='Random Forest: GDP of the country is  $ {}'.format(output), prediction_text2='Linear Regression: GDP of the country is  $ {}'.format(output1))
@app.route('/country',methods=['POST'])
def country():
    '''
    For rendering results on HTML GUI
    '''
    
    text=request.form.get("country")
   
    return render_template('index.html', prediction_text3=text)
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)