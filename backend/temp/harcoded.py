import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import logic  # Import the entire logic.py module

app = Flask(__name__)
CORS(app)

@app.route('/api/save_parameters', methods=['POST'])
def save_parameters():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid data'}), 400
    
    tickers = [
        'USDEUR Curncy', 'USDJPY Curncy', 'USDCHF Curncy', 'USDGBP Curncy', 'USDCAD Curncy',
        'USDAUD Curncy', 'USDNZD Curncy', 'USDNOK Curncy', 'USDSEK Curncy', 'USDMXN Curncy',
        'USDCNH Curncy', 'KWN+1M Curncy',
        'IRN+1M Curncy', 'USDTHB CMPN Curncy', 'USDSGD Curncy', 'NTN+1M Curncy',
        'PPN+1M Curncy', 'USDTWD REGN Curncy', 'CCN+3M Curncy', 'XAU Curncy', 'NQ1 Index',
    ]

    groups = {
        'Group 1': [(0, -8.85, 'FX'), (1, 2.55, 'FX'), (2, 14.00, 'FX'), (3, -20.55, 'FX')],
        'Group 2': [(4, -5.75, 'FX'), (5, -18.60, 'FX'), (6, 10.00, 'FX')],
        'Group 3': [(7, 4.00, 'FX'), (8, 11.50, 'FX')],
        'Group 4': [(9, 2.00, 'FX')],
        'Group 5': [
            (10, 56.00, 'FX'), (11, -5.50, 'FX'), (12, -43.50, 'FX'), (13, 5.50, 'FX'), 
            (14, 32.25, 'FX'), (15, -8.50, 'FX'), (16, 1.00, 'FX'), 
            (17, 4.50, 'FX'), (18, 10.00, 'FX'), (19, 4, 'FX'),
            (20, 60, 'Equity (ETF, Index, Futures)')
        ],
    }

    fields = ['Last Price']
    periods = ['100 days', '3 months']
    end_date = '2024-07-12'
    confidence_levels = [0.90, 0.95, 0.99]
    z_scores = {'90%': 1.28, '95%': 1.65, '99%': 2.33}

    # Run the main logic function
    results = logic.main(tickers, fields, periods, end_date, confidence_levels, z_scores, groups)

    # Convert the groups dictionary to a DataFrame
    group_data = []
    for group_name, items in groups.items():
        for item in items:
            ticker_index, size, item_type = item
            group_data.append([group_name, tickers[ticker_index], size, item_type])

    df_groups = pd.DataFrame(group_data, columns=['Group', 'Ticker', 'Size', 'Type'])

    # Create DataFrames for tickers, periods, and end_date
    df_tickers = pd.DataFrame({'Tickers': tickers})
    df_periods = pd.DataFrame({'Periods': periods})
    df_end_date = pd.DataFrame({'End Date': [end_date]})

    # Ensure the outputs directory exists
    outputs_dir = os.path.join(os.getcwd(), 'outputs')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    # Save the file to the outputs directory
    save_path = os.path.join(outputs_dir, 'parameters.xlsx')
    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        df_tickers.to_excel(writer, sheet_name='Tickers', index=False)
        df_periods.to_excel(writer, sheet_name='Periods', index=False)
        df_end_date.to_excel(writer, sheet_name='End Date', index=False)
        df_groups.to_excel(writer, sheet_name='Groups', index=False)

    return jsonify({'message': 'File saved successfully', 'path': save_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)