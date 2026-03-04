import numpy as np
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
 
# Initialize Flask app
app = Flask(__name__)
 
# Load and train model (runs once when server starts)
iris = load_iris()
X = iris.data
Y = iris.target
 
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)
 
model = GaussianNB()
model.fit(X_train, Y_train)
 
# Home route
@app.route("/")
def home():
    return "Iris Naive Bayes Model is Running!"
 
# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
 
    features = np.array(data).reshape(1, -1)
    prediction = model.predict(features)[0]
 
    return jsonify({
        "prediction": int(prediction),
        "class": iris.target_names[prediction]
    })
 
# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)