from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        total_sqft = float(data['total_sqft'])
        house_age = float(data['house_age'])
        renovation_age = float(data['renovation_age'])
        bedroom_to_bathroom_ratio = float(data['bedroom_to_bathroom_ratio'])
        log_price = float(data['log_price'])
        sqrt_sqft_living = float(data['sqrt_sqft_living'])
        has_basement = int(data['has_basement'])
        is_waterfront = int(data['is_waterfront'])
        has_view = int(data['has_view'])

        features = [total_sqft, house_age, renovation_age, bedroom_to_bathroom_ratio, log_price, sqrt_sqft_living, has_basement, is_waterfront, has_view]
        predicted_log_price = model.predict([features])[0]
        predicted_price = np.expm1(predicted_log_price)

        return render_template('index.html', prediction_result=predicted_price)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
     app.run(host='0.0.0.0',port=8080)