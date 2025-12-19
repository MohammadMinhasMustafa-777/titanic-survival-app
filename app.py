from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the full pipeline and extract ONLY the trained RandomForest (bypasses imputer)
pipeline = joblib.load('titanic_model_pipeline.pkl')
model = pipeline.named_steps['model']  # This is your trained RF

# Hardcoded medians (standard Titanic values)
AGE_MEDIAN = 28.0
FARE_MEDIAN = 14.4542

# HTML form with retained values
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Titanic Survival Predictor</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; }
        h1 { text-align: center; }
        form { background: #f0f0f0; padding: 20px; border-radius: 10px; }
        label { display: block; margin: 10px 0 5px; }
        input, select { width: 100%; padding: 8px; box-sizing: border-box; }
        button { margin-top: 20px; padding: 10px; background: #007bff; color: white; border: none; width: 100%; font-size: 18px; cursor: pointer; }
        .result { margin-top: 30px; font-size: 24px; text-align: center; padding: 20px; border-radius: 10px; }
        .survived { background: #d4edda; color: #155724; }
        .not-survived { background: #f8d7da; color: #721c24; }
        .error { background: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <h1>Titanic Survival Predictor</h1>
    <form method="post">
        <label>Pclass:</label>
        <select name="Pclass">
            <option value="1" {{ 'selected' if pclass == 1 else '' }}>1st Class</option>
            <option value="2" {{ 'selected' if pclass == 2 else '' }}>2nd Class</option>
            <option value="3" {{ 'selected' if pclass == 3 else '' }}>3rd Class</option>
        </select>

        <label>Sex:</label>
        <select name="Sex">
            <option value="male" {{ 'selected' if sex == 'male' else '' }}>Male</option>
            <option value="female" {{ 'selected' if sex == 'female' else '' }}>Female</option>
        </select>

        <label>Age:</label>
        <input type="number" name="Age" value="{{ age }}" step="1">

        <label>SibSp:</label>
        <input type="number" name="SibSp" value="{{ sibsp }}" min="0">

        <label>Parch:</label>
        <input type="number" name="Parch" value="{{ parch }}" min="0">

        <label>Fare:</label>
        <input type="number" name="Fare" value="{{ fare }}" step="0.01">

        <label>Embarked:</label>
        <select name="Embarked">
            <option value="S" {{ 'selected' if embarked == 'S' else '' }}>Southampton</option>
            <option value="C" {{ 'selected' if embarked == 'C' else '' }}>Cherbourg</option>
            <option value="Q" {{ 'selected' if embarked == 'Q' else '' }}>Queenstown</option>
        </select>

        <button type="submit">Predict Survival</button>
    </form>

    {% if result %}
    <div class="result {{ result.class }}">
        <strong>{{ result.message }}</strong><br>
        Confidence: {{ result.confidence }}
    </div>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    # Default values for form
    pclass = 3
    sex = 'male'
    age = 25
    sibsp = 0
    parch = 0
    fare = 7.25
    embarked = 'S'

    result = None
    if request.method == 'POST':
        try:
            # Get submitted values
            pclass = int(request.form['Pclass'])
            sex = request.form['Sex']
            age = float(request.form.get('Age') or AGE_MEDIAN)
            sibsp = int(request.form.get('SibSp') or 0)
            parch = int(request.form.get('Parch') or 0)
            fare = float(request.form.get('Fare') or FARE_MEDIAN)
            embarked = request.form['Embarked']

            # Manual preprocessing (exact match to your pipeline)
            df = pd.DataFrame([{
                'Pclass': pclass,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'Sex': sex,
                'Embarked': embarked
            }])

            # Manual imputation
            df['Age'].fillna(AGE_MEDIAN, inplace=True)
            df['Fare'].fillna(FARE_MEDIAN, inplace=True)

            # Manual one-hot (drop='first')
            df['Sex_male'] = (df['Sex'] == 'male').astype(int)
            df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
            df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
            df = df.drop(['Sex', 'Embarked'], axis=1)

            # Exact column order as training
            df = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]

            # Predict with the extracted RF model
            prediction = int(model.predict(df)[0])
            confidence = model.predict_proba(df)[0][prediction]

            result = {
                'class': 'survived' if prediction == 1 else 'not-survived',
                'message': 'You Survived!' if prediction == 1 else 'You Did Not Survive',
                'confidence': f"{confidence:.1%}"
            }
        except Exception as e:
            result = {
                'class': 'error',
                'message': f'Error: {str(e)}',
                'confidence': ''
            }

    return render_template_string(HTML_TEMPLATE, result=result,
                                  pclass=pclass, sex=sex, age=age,
                                  sibsp=sibsp, parch=parch, fare=fare, embarked=embarked)

if __name__ == '__main__':
    app.run()