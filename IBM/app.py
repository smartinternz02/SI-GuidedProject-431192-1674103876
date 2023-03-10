from flask import Flask, render_template, request # Flask is a application
import requests
# used to run/serve our application
# request is used to access the file which is uploaded by the user in out application
# render_template is used for rendering the html pages
import pickle # pickle is used for serializing and de-serializing Python object structures
from sklearn.preprocessing import StandardScaler

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "lRg71Qhee3BGRe4qxZ07lHIGEbFSKH7HDuoDxDTYOgx7"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
app = Flask(__name__) # our flask app


@app.route('/')  # rendering the html template
def home():
    return render_template('home.html')


@app.route('/predict')  # rendering the html template
def index():
    return render_template("predict.html")


@app.route('/pred', methods=['GET', 'POST'])  # route for our prediction
def predict():
    Region = request.form['Region']
    Population = request.form['Population']
    Area = request.form['Area']
    Pop = request.form['Pop']
    Coastline = request.form['Coastline']
    Phones = request.form['Phones']
    Arable = request.form['Arable']
    Crops = request.form['Crops']
    Climate = request.form['Climate']
    Birthrate = request.form['Birthrate']
    Deathrate = request.form['Deathrate']
    Agriculture = request.form['Agriculture']
    Industry = request.form['Industry']
    Service = request.form['Service']
    total = [
        [Region, Population, Area, Pop, Coastline, Phones, Arable, Crops, Climate, Birthrate, Deathrate, Agriculture,
         Industry, Service]]
    print(total)
    payload_scoring = {"input_data": [{"field": [["Region", "Population", "Area", "Pop", "Coastline",  "Phones", "Arable", "Crops", "Climate", "Birthrate", "Deathrate", "Agriculture", "Industry", 'Service']], "values": total}]}
    # print(total)
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/50daad77-6447-401d-b535-522274ae267d/predictions?version=2023-02-13', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    # predictions = response_scoring.json()
    pred = response_scoring.json()
    print(pred)
    predictio = pred['predictions'][0]['values'][0][0]
    
    return render_template('gdp_pred.html', prediction=predictio)


if __name__ == '__main__':
    app.run(debug=True)# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

