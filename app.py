import numpy as np
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Load dataset
iris = load_iris()
X = iris.data
Y = iris.target

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

# Train model
model = GaussianNB()
model.fit(X_train, Y_train)

# Calculate accuracy
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# Home route (Now returns structured output)
@app.route("/")
def home():
    return jsonify({
        "message": "Iris Naive Bayes Model API is Running",
        "model": "Gaussian Naive Bayes",
        "accuracy": round(float(accuracy), 4),
        "expected_input_format": {
            "features": [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width"
            ]
        }
    })

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]

        if len(data) != 4:
            return jsonify({
                "error": "Please provide exactly 4 feature values."
            }), 400

        features = np.array(data).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        return jsonify({
            "input_features": data,
            "prediction_index": int(prediction),
            "predicted_class": iris.target_names[prediction],
            "prediction_confidence": {
                iris.target_names[i]: round(float(probability[i]), 4)
                for i in range(len(probability))
            }
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)