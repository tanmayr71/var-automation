import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logic  # Import the entire logic.py module

app = Flask(__name__)
CORS(app)

@app.route('/api/save_parameters', methods=['POST'])
def save_parameters():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid data'}), 400
    
    tickers = data['tickers']
    periods = data['periods']
    end_date = data['endDate']
    groups = data['groups']

    # Define the confidence levels and Z-scores
    confidence_levels = [0.90, 0.95, 0.99]
    z_scores = {'90%': 1.28, '95%': 1.65, '99%': 2.33}

    # Call the main logic function
    results, file_path = logic.main(tickers, ['Last Price'], periods, end_date, confidence_levels, z_scores, groups)

    return jsonify({'message': 'File saved successfully', 'results_path': file_path})

@app.route('/api/download', methods=['GET'])
def download_file():
    file_path = request.args.get('file_path')
    if file_path:
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File path not provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
