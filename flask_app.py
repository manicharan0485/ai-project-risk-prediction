from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        df = pd.DataFrame(data)
        scaler = joblib.load('scaler.pkl')
        polynomial_features = joblib.load('polynomial_features.pkl')
        model = joblib.load('stacking_model.pkl')
        
        X_poly = polynomial_features.transform(df)
        X_scaled = scaler.transform(X_poly)
        
        predictions = model.predict(X_scaled)
        
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
