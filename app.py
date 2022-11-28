from flask import Flask, render_template, request, redirect, url_for
import os
import csv
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__,template_folder='templates')

model_regresi = joblib.load('model_regresi.pkl')
model_klasifikasi = joblib.load('model_klasifikasi.pkl')

@app.route("/")
def root():
    return render_template("index.html")

@app.route("/regresi", methods=['GET', 'POST'])
def prediksi_regresi():
    uploaded_file = request.files['x_regresi']
    filepath = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
    uploaded_file.save(filepath)
    with open(filepath) as file:
        data=pd.read_csv(file, header=None)

    prediction_results = model_regresi.predict(data)

    with np.printoptions(threshold=np.inf):
        str_data = str(prediction_results)
        
    return render_template('index.html', hasil_regresi=str_data)

@app.route("/klasifikasi", methods=['GET', 'POST'])
def prediksi_klasifikasi():
    data = []

    uploaded_file = request.files['x_klasifikasi']
    filepath = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
    uploaded_file.save(filepath)
    with open(filepath) as file:
        csv_file = csv.reader(file)
        for row in csv_file:
            data.append(row)
    
    prediction_results = model_klasifikasi.predict(data)
    
    str_data = str(prediction_results)
        
    return render_template('index.html', hasil_klasifikasi=str_data)

app.config['FILE_UPLOADS'] = "uploads"

if __name__ == '__main__':
    app.run(debug=True)