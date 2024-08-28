from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model and preprocessor
model = joblib.load('models/model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        dep_time = request.form['Dep_Time']
        arrival_time = request.form['Arrival_Time']
        source = request.form['Source']
        destination = request.form['Destination']
        stops = request.form['stops']
        airline = request.form['airline']

        # Parse departure time
        dep_time_obj = pd.to_datetime(dep_time, format='%Y-%m-%dT%H:%M')
        dep_hour = dep_time_obj.hour
        dep_minute = dep_time_obj.minute

        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'Journey_day': [dep_time_obj.day],
            'Journey_month': [dep_time_obj.month],
            'Dep_hour': [dep_hour],
            'Dep_min': [dep_minute],
            'Source': [source],
            'Destination': [destination],
            'Airline': [airline],
            'Total_Stops': [stops],
        })

        # Apply preprocessing (encoding categorical features)
        input_data_transformed = preprocessor.transform(input_data)

        # Ensure the transformed data is 2D
        if input_data_transformed.ndim == 1:
            input_data_transformed = input_data_transformed.reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data_transformed)
        prediction = prediction.flatten()[0]

        return render_template('index.html', prediction_text=f'Predicted Price: ₹{prediction:.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
