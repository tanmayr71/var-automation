# import pandas as pd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os

# app = Flask(__name__)
# CORS(app)

# @app.route('/api/save_parameters', methods=['POST'])
# def save_parameters():
#     data = request.json
#     if not data:
#         return jsonify({'error': 'Invalid data'}), 400
    
#     tickers = data['tickers']
#     periods = data['periods']
#     end_date = data['endDate']
#     groups = data['groups']

#     # Convert the groups dictionary to a DataFrame
#     group_data = []
#     for group_name, items in groups.items():
#         for item in items:
#             ticker_index, size, item_type = item
#             group_data.append([group_name, tickers[ticker_index], size, item_type])

#     df_groups = pd.DataFrame(group_data, columns=['Group', 'Ticker', 'Size', 'Type'])

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