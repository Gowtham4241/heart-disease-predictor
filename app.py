from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("❌ Model file not found. Please train and save the model first.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        Age = float(request.form['age'])
        cigsPerDay = float(request.form['cigsPerDay'])
        Diabetes = float(request.form['diabetes'])
        TotalCholastrol = float(request.form['totChol'])
        DiaBP = float(request.form['DiaBP'])
        BMI = float(request.form['BMI'])
        glucose = float(request.form['glucose'])

        # Create feature array
        features = np.array([[Age, cigsPerDay, Diabetes, TotalCholastrol, DiaBP, BMI, glucose]])

        # Check if model is loaded
        if model is None:
            return render_template('index.html', prediction_text="Model not loaded. Please check 'model.pkl'.")

        # Make prediction
        prediction = model.predict(features)[0]
        result = "RISK ✅" if prediction == 1 else "NO RISK ❌"

        return render_template('index.html', prediction_text=f'RISK of coronary heart disease: {result}')

    except ValueError as e:
        return render_template('index.html', prediction_text=f'ValueError: {str(e)}')
    except KeyError as e:
        return render_template('index.html', prediction_text=f'KeyError: {str(e)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
