from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import traceback

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            print(" Step 1: Collecting form data...")
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            print(" Step 2: Converting to DataFrame...")
            pred_df = data.get_data_as_data_frame()
            print(" Input DataFrame:")
            print(pred_df)

            print(" Step 3: Loading model & preprocessor...")
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            print(" Step 4: Prediction done.")
            return render_template('home.html', results=results[0])

        except Exception as e:
            error_msg = traceback.format_exc()
            print("Error caught:\n", error_msg)
            return f"<h2>Server Error</h2><pre>{error_msg}</pre>"
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        


