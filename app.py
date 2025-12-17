from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
model = joblib.load('titanic_model_pipeline.pkl')

@app.route('/')
def home():
    return '''
    <h1>Titanic Survival Predictor</h1>
    <p>Send POST request to /predict with JSON:</p>
    <pre>{
  "Pclass": 3,
  "Sex": "male",
  "Age": 25,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}</pre>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].max()
    
    return jsonify({
        'Survived': int(prediction),
        'Confidence': round(probability, 3)
    })

if __name__ == '__main__':
    app.run()