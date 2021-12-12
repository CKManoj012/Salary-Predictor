from flask import Flask,redirect,url_for,request,render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
sc = pickle.load(open("standardScalar.pkl","rb"))

@app.route("/")
def home():
    return render_template('input.html')
   
@app.route("/predict",methods=["GET","POST"])
def prediction():
    int_features = [float(x) for x in request.form.values()]   
    
    
    trans_features = sc.transform([[float(x) for x in request.form.values()]])

    predictions = model.predict(trans_features)
    return render_template("input.html",salary = "The expected Salary is  ${}".format(round(predictions[0])))   
    
  #  sc = StandardScaler()
  #  predictions = model.predict(final_pred)

    #return render_template("input.html",salary = "The expected Salary is  ${}".format(round(predictions[0],0)))   


if __name__ == "__main__":
        app.run(debug = True)

