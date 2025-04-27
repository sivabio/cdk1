import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
cdk_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@cdk_app.route("/")
def Home():
    return render_template("index.html")

@cdk_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    return render_template("after.html", data=prediction)

if __name__ == "__main__":
    cdk_app.run(debug=True)