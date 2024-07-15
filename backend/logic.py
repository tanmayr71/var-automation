import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import pandas as pd
from xbbg import blp
import numpy as np
from scipy.stats import skew, kurtosis
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt

import logging
import os

from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def adjust_column_widths(worksheet, dataframe):
    for col_num, col_name in enumerate(dataframe.columns):
        max_length = max(
            dataframe[col_name].astype(str).map(len).max() if len(dataframe[col_name]) > 0 else 0,
            len(col_name)
        ) + 2  # Add a little extra space
        worksheet.set_column(col_num, col_num, max_length)

def plot_dd_tuw_portfolio(portfolio_data, period, show_dd=True, show_tuw=False, save=False, save_dir=None):
    pnl_series = portfolio_data['portfolio_changes']
    hwm_series = portfolio_data['portfolio_hwm']
    hwm_indices = portfolio_data['portfolio_hwm_indices']
    dd_series = portfolio_data['portfolio_dd']
    tuw_series = portfolio_data['portfolio_tuw']
    
    # Ensure the series indices are of type pd.Timestamp
    pnl_series.index = pd.to_datetime(pnl_series.index)
    hwm_series.index = pd.to_datetime(hwm_series.index)
    dd_series.index = pd.to_datetime(dd_series.index)
    tuw_series.index = pd.to_datetime(tuw_series.index)
    
    # Create DataFrame for plotting
    cum_pnl_series = pnl_series.expanding().sum()
    df0 = cum_pnl_series.to_frame('pnl')
    df0['hwm'] = hwm_series
    
    # Plot PnL series and HWM
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df0.index, df0['pnl'], label='Cumulative PnL', color='blue')
    ax1.plot(df0.index, df0['hwm'], label='HWM', color='green', linestyle='--')
    
    # Highlight the drawdown periods
    if show_dd:
        for i in range(len(dd_series)):
            start = pd.Timestamp(hwm_indices[i])
            end = pd.Timestamp(hwm_indices[i + 1]) if i + 1 < len(hwm_indices) else df0.index[-1]
            drawdown_period = df0.loc[start:end]
            ax1.fill_between(drawdown_period.index, drawdown_period['pnl'], drawdown_period['hwm'], color='red', alpha=0.3)
    
    # Plot Time Under Water (TUW)
    if show_tuw:
        ax2 = ax1.twinx()
        ax2.plot(tuw_series.index, tuw_series.values, label='TUW', color='orange', marker='o', linestyle='-', alpha=0.6)
        ax2.set_ylabel('Time Under Water (Days)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
    
    # Plot Drawdowns (DD)
    if show_dd:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
        ax3.plot(dd_series.index, dd_series.values, label='Drawdowns (DD)', color='red', marker='o', linestyle='-', alpha=0.6)
        ax3.set_ylabel('Drawdown Value', color='red')
        ax3.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    if show_tuw:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    if show_dd:
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines += lines3
        labels += labels3
    
    ax1.legend(lines, labels, loc='best')
    
    ax1.set_title(f'PnL, High Water Marks (HWM), Drawdowns (DD), and Time Under Water (TUW) for Portfolio ({period})')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PnL')

    if save and save_dir:
        image_path = os.path.join(save_dir, f'portfolio_dd_tuw_{period}.png')
        fig.savefig(image_path)
        plt.close(fig)  # Close the plot to free up memory
        return image_path
    
    plt.show()

def plot_tuw_portfolio(portfolio_data, period, save=False, save_dir=None):
    tuw_series = portfolio_data['portfolio_tuw']
    
    # Plot Time Under Water (TUW)
    fig, ax3 = plt.subplots(figsize=(14, 7))
    ax3.bar(tuw_series.index, tuw_series.values, label='TUW', color='orange', alpha=0.6)
    ax3.set_ylabel('Time Under Water (Days)', color='orange')
    ax3.set_title(f'Time Under Water (TUW) for Portfolio ({period})')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Days')
    ax3.legend()

    if save and save_dir:
        image_path = os.path.join(save_dir, f'portfolio_tuw_{period}.png')
        fig.savefig(image_path)
        plt.close(fig)  # Close the plot to free up memory
        return image_path
    
    plt.show()

def save_dd_tuw_portfolio_plot(portfolio_data, period, save_dir):
    return plot_dd_tuw_portfolio(portfolio_data, period, save=True, save_dir=save_dir)

def save_tuw_portfolio_plot(portfolio_data, period, save_dir):
    return plot_tuw_portfolio(portfolio_data, period, save=True, save_dir=save_dir)

# Define a function to format and insert images
def insert_image_with_formatting(sheet, image_path, row, col, width=400, height=300):
    sheet.insert_image(row, col, image_path, {'x_scale': width / 800, 'y_scale': height / 600, 'positioning': 1})

def fill_summary_data(sheet, formats, all_results, periods, measures, confidence_levels, start_row, unique_dir):
    groups_list = list(all_results[periods[0]]['group_results'].keys())
    sheet.write(start_row, 0, 'Portfolio', formats['left_align'])  # Add portfolio row
    for row, group in enumerate(groups_list, start=start_row + 1):  # Adjust start row for portfolio
        sheet.write(row, 0, group, formats['left_align'])

    # Fill in the data for the portfolio and each group
    last_data_row = start_row + len(groups_list) + 1  # Calculate the last data row
    for period_index, period in enumerate(periods):
        period_offset = period_index * len(measures) + 1
        portfolio_data = all_results[period]['portfolio_results']
        sheet.write(start_row, period_offset, portfolio_data['portfolio_mean'], formats['data'])
        sheet.write(start_row, period_offset + 1, portfolio_data['portfolio_std_dev'], formats['data'])
        sheet.write(start_row, period_offset + 2, portfolio_data['portfolio_skewness'], formats['data'])
        sheet.write(start_row, period_offset + 3, portfolio_data['portfolio_kurtosis'], formats['data'])
        for i, conf in enumerate(confidence_levels):
            sheet.write(start_row, period_offset + 4 + i, portfolio_data['portfolio_VaR'][f'{int(conf*100)}%'], formats['data'])

    # Fill in the data for each group and period
    for period_index, period in enumerate(periods):
        period_offset = period_index * len(measures) + 1
        for row, group in enumerate(groups_list, start=start_row + 1):  # Adjust start row for portfolio
            group_data = all_results[period]['group_results'][group]
            sheet.write(row, period_offset, group_data['group_mean'], formats['data'])
            sheet.write(row, period_offset + 1, group_data['group_std_dev'], formats['data'])
            sheet.write(row, period_offset + 2, group_data['group_skewness'], formats['data'])
            sheet.write(row, period_offset + 3, group_data['group_kurtosis'], formats['data'])
            for i, conf in enumerate(confidence_levels):
                sheet.write(row, period_offset + 4 + i, group_data['group_VaR'][f'{int(conf*100)}%'], formats['data'])

    # Insert the images into the summary sheet below the data table
    image_insert_row = last_data_row + 3  # Adjust the row to place the images below the data
    for period_index, period in enumerate(periods):
        period_offset = period_index * len(measures) + 1
        portfolio_data = all_results[period]['portfolio_results']
        
        # Generate and save the plots
        dd_tuw_image_path = save_dd_tuw_portfolio_plot(portfolio_data, period, unique_dir)
        tuw_image_path = save_tuw_portfolio_plot(portfolio_data, period, unique_dir)
        
        # Insert the images into the summary sheet
        insert_image_with_formatting(sheet, dd_tuw_image_path, image_insert_row, period_offset, width=400, height=300)
        insert_image_with_formatting(sheet, tuw_image_path, image_insert_row + 18, period_offset, width=400, height=300)  # Adjust row for second image

def write_summary_header(sheet, formats, periods, measures, start_row):
    sheet.merge_range(start_row, 0, start_row + 1, 0, 'Group', formats['header'])
    col = 1
    for period in periods:
        sheet.merge_range(start_row, col, start_row, col + len(measures) - 1, period, formats['header'])
        for i, measure in enumerate(measures):
            sheet.write(start_row + 1, col + i, measure, formats['sub_header'])
        col += len(measures)

def write_sizer_overview(sheet, formats, groups, tickers):
    sheet.merge_range('A1:D1', 'Sizer Overview', formats['title'])
    sheet.write('A2', 'Ticker', formats['header'])
    sheet.write('B2', 'Weight', formats['header'])
    sheet.write('C2', 'Group', formats['header'])
    sheet.write('D2', 'Type', formats['header'])
    
    sizer_data = []
    for group_name, group_info in groups.items():
        for (idx, size, ticker_type) in group_info:
            ticker = tickers[idx]
            sizer_data.append([ticker, size, group_name, ticker_type])
    
    sizer_df = pd.DataFrame(sizer_data, columns=['Ticker', 'Weight', 'Group', 'Type'])
    for row, (ticker, weight, group, ticker_type) in enumerate(sizer_df.values, start=3):
        row_format = formats['alt_row'] if row % 2 == 0 else formats['left_align']
        sheet.write(row, 0, ticker, row_format)
        sheet.write(row, 1, weight, formats['data'])
        sheet.write(row, 2, group, row_format)
        sheet.write(row, 3, ticker_type, row_format)
    
    return sizer_df

def define_formats(workbook):
    formats = {
        'header': workbook.add_format({'bold': True, 'bg_color': '#F7DC6F', 'border': 1, 'align': 'center', 'font_size': 12}),
        'sub_header': workbook.add_format({'bold': True, 'bg_color': '#F9E79F', 'border': 1, 'align': 'center', 'font_size': 11}),
        'data': workbook.add_format({'border': 1, 'num_format': '#,##0.00', 'font_size': 10}),
        'title': workbook.add_format({'bold': True, 'font_size': 14, 'align': 'center', 'bg_color': '#D5DBDB'}),
        'ticker': workbook.add_format({'border': 1, 'text_wrap': True, 'font_size': 10}),
        'left_align': workbook.add_format({'border': 1, 'align': 'left', 'font_size': 10}),
        'alt_row': workbook.add_format({'bg_color': '#F2F2F2', 'border': 1, 'font_size': 10}),
        'note': workbook.add_format({'italic': True, 'font_color': '#555555', 'font_size': 10})  # Added note format
    }
    return formats

# def create_directory(file_prefix):
#     unique_dir = f"{file_prefix}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
#     os.makedirs(unique_dir, exist_ok=True)
#     return unique_dir

def create_directory(file_prefix):
    outputs_dir = os.path.join(os.getcwd(), 'outputs')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    unique_dir = os.path.join(outputs_dir, f"{file_prefix}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}")
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

def write_beta_analysis_header(sheet, formats, periods, benchmarks):
    # Write the first row with periods spanning multiple columns
    col = 1  # Start from the second column (first column is for Group/Portfolio)
    for period in periods:
        sheet.merge_range(0, col, 0, col + len(benchmarks) - 1, period, formats['header'])
        col += len(benchmarks)

    # Write the second row with benchmark names
    col = 1  # Start from the second column
    for _ in periods:
        for benchmark in benchmarks:
            sheet.write(1, col, benchmark[0], formats['sub_header'])  # Write only the benchmark name
            col += 1

def fill_beta_analysis_data(sheet, formats, all_results, benchmarks, display_in_millions=True):
    row = 2  # Start from the third row
    groups_set = set()

    # Collect all groups
    for period, period_results in all_results.items():
        groups_set.update(period_results['group_results'].keys())

    groups_list = list(groups_set)

    # Write the Group/Portfolio names in the first column
    for row_idx, group in enumerate(groups_list):
        sheet.write(row + row_idx, 0, group, formats['left_align'])
    sheet.write(row + len(groups_list), 0, 'Portfolio', formats['left_align'])

    for period_idx, period in enumerate(all_results.keys()):
        period_results = all_results[period]
        col = 1 + period_idx * len(benchmarks)  # Calculate the start column for each period
        for benchmark in benchmarks:
            for row_idx, group in enumerate(groups_list):
                group_beta_analysis = period_results['group_results'].get(group, {}).get('beta_analysis', {})
                group_beta = group_beta_analysis.get('beta', {})
                beta_value = group_beta.get(benchmark, None)  # Use the full benchmark name
                if beta_value is not None:
                    if display_in_millions:
                        beta_value /= 10**6
                    sheet.write(row + row_idx, col, float(beta_value), formats['data'])
            portfolio_beta_analysis = period_results['portfolio_results']['beta_analysis']
            portfolio_beta = portfolio_beta_analysis.get('beta', {})
            beta_value = portfolio_beta.get(benchmark, None)  # Use the full benchmark name
            if beta_value is not None:
                if display_in_millions:
                    beta_value /= 10**6
                sheet.write(row + len(groups_list), col, float(beta_value), formats['data'])
            col += 1

    # Calculate the row index for the note
    note_row = row + len(groups_list) + 1
    if display_in_millions:
        sheet.write(note_row, 0, "Note: Beta values are displayed in millions.", formats['note'])

def write_beta_analysis_to_sheet(workbook, formats, all_results):
    # Define the benchmarks
    benchmarks = [
        ('DXY Curncy', 'Last Price'), 
        ('USDCNH Curncy', 'Last Price'), 
        ('SPX Index', 'Last Price'), 
        ('CO1 Comdty', 'Last Price'), 
        ('USOSFR10 CMPN Curncy', 'Last Price'), 
        ('USOSFR10 CMPT Curncy', 'Last Price')
    ]
    
    # Create a new sheet for beta analysis
    beta_sheet = workbook.add_worksheet('Beta Analysis')
    
    # Write headers and fill data for beta analysis
    periods = list(all_results.keys())
    write_beta_analysis_header(beta_sheet, formats, periods, benchmarks)
    fill_beta_analysis_data(beta_sheet, formats, all_results, benchmarks)
    
    # Adjust column widths for beta analysis sheet
    adjust_column_widths(beta_sheet, pd.DataFrame(columns=['Group/Portfolio'] + [b[0] for b in benchmarks] * len(periods)))

def save_results_to_file(all_results, groups, tickers, fields, confidence_levels, file_prefix='var_results'):
    # Create a unique directory for this call
    unique_dir = create_directory(file_prefix)
    file_name = f"{unique_dir}/{file_prefix}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.xlsx"
    
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        formats = define_formats(workbook)
        
        # Create a summary sheet
        summary_sheet = workbook.add_worksheet('Summary')
        
        # Write Sizer Overview at the top of the summary page
        sizer_df = write_sizer_overview(summary_sheet, formats, groups, tickers)
        
        # Leave some space after the Sizer Overview table
        start_row = len(sizer_df) + 5
        
        # Define column structure
        periods = list(all_results.keys())
        measures = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis'] + [f'VaR {int(conf*100)}%' for conf in confidence_levels]
        
        # Write header for the summary sheet
        write_summary_header(summary_sheet, formats, periods, measures, start_row)
        
        # Fill in the data for the portfolio and each group
        fill_summary_data(summary_sheet, formats, all_results, periods, measures, confidence_levels, start_row + 2, unique_dir)  # Adjust start_row for filling data
        
        # Freeze panes for better navigation
        # summary_sheet.freeze_panes(start_row + 2, 1)
        
        # General information and Sizer Sheet per period
        for period, results in all_results.items():
            sheet_name = f'Sizer_{period}'
            sizer_sheet = workbook.add_worksheet(sheet_name)
            
            # General information
            general_info = {
                'Start Date': results['start_date'],
                'End Date': results['end_date'],
                'Period': period,
                'Field': fields[0],
                'Number of Data Points': len(results['historical_data'])
            }
            df_info = pd.DataFrame(list(general_info.items()), columns=['Description', 'Value'])
            df_info.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
            
            # Sizer Sheet
            sizer_data = []
            for group_name, group_info in results['group_results'].items():
                for (idx, size, ticker_type) in groups[group_name]:
                    ticker = tickers[idx]
                    sizer_data.append([ticker, size, group_name, ticker_type])
            
            sizer_df = pd.DataFrame(sizer_data, columns=['Ticker', 'Weight', 'Group', 'Type'])
            sizer_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(df_info) + 3)
            
            # Add measures calculated for each group
            measures_data = []
            for group_name, group_data in results['group_results'].items():
                measures_data.append([
                    group_name,
                    group_data['group_mean'],
                    group_data['group_std_dev'],
                    group_data['group_skewness'],
                    group_data['group_kurtosis']
                ] + [group_data['group_VaR'][f'{int(conf*100)}%'] for conf in confidence_levels])
            
            measures_df = pd.DataFrame(measures_data, columns=['Group', 'Mean', 'Std Dev', 'Skewness', 'Kurtosis'] + [f'VaR {conf}%' for conf in confidence_levels])
            measures_df.to_excel(writer, sheet_name=sheet_name, startrow=len(df_info) + len(sizer_df) + 7, index=False)
            
            # Styling the Sizer sheet
            sizer_sheet.write('A1', 'General Information', formats['title'])
            for col_num, value in enumerate(df_info.columns.values):
                sizer_sheet.write(1, col_num, value, formats['header'])
            for row_num, _ in enumerate(df_info.values):
                for col_num in range(len(df_info.columns)):
                    sizer_sheet.write(2 + row_num, col_num, df_info.iat[row_num, col_num], formats['left_align'])
            
            sizer_sheet.write(len(df_info) + 2, 0, 'Sizer Overview', formats['title'])
            for col_num, value in enumerate(sizer_df.columns.values):
                sizer_sheet.write(len(df_info) + 3, col_num, value, formats['header'])
            for row_num, _ in enumerate(sizer_df.values):
                for col_num in range(len(sizer_df.columns)):
                    sizer_sheet.write(len(df_info) + 4 + row_num, col_num, sizer_df.iat[row_num, col_num], formats['ticker'] if col_num == 0 else formats['data'])

            sizer_sheet.write(len(df_info) + len(sizer_df) + 6, 0, 'Group Measures', formats['title'])
            for col_num, value in enumerate(measures_df.columns.values):
                sizer_sheet.write(len(df_info) + len(sizer_df) + 7, col_num, value, formats['header'])
            for row_num, _ in enumerate(measures_df.values):
                for col_num in range(len(measures_df.columns)):
                    sizer_sheet.write(len(df_info) + len(sizer_df) + 8 + row_num, col_num, measures_df.iat[row_num, col_num], formats['data'])
            
            # Adjust column widths
            adjust_column_widths(sizer_sheet, sizer_df)
            adjust_column_widths(sizer_sheet, measures_df)
            adjust_column_widths(sizer_sheet, df_info)
            adjust_column_widths(summary_sheet, sizer_df)

            # Adjust header row heights for better readability
            sizer_sheet.set_row(len(df_info) + 3, 30)  # Adjust Sizer Overview header row height
            measures_header_row = len(df_info) + len(sizer_df) + 7
            sizer_sheet.set_row(measures_header_row, 30)  # Adjust Group Measures header row height

            # Calculate column to place the images to the right of the data
            image_insert_col = len(measures_df.columns) + 2
        
            # Insert the images into the Sizer sheet to the right of the data
            dd_tuw_image_path = save_dd_tuw_portfolio_plot(results['portfolio_results'], period, unique_dir)
            tuw_image_path = save_tuw_portfolio_plot(results['portfolio_results'], period, unique_dir)
            insert_image_with_formatting(sizer_sheet, dd_tuw_image_path, 0, image_insert_col, width=400, height=300)
            insert_image_with_formatting(sizer_sheet, tuw_image_path, 18, image_insert_col, width=400, height=300)

        # Adjust summary sheet column widths
        for col_num in range(summary_sheet.dim_colmax + 1):
            summary_sheet.set_column(col_num, col_num, 15)

        # Adjust header row heights for better readability
        summary_sheet.set_row(start_row, 30)  # Adjust summary header row height
        summary_sheet.set_row(start_row + 1, 30)  # Adjust sub-header row height

        # Call the function to handle all tasks related to the beta sheet
        write_beta_analysis_to_sheet(workbook, formats, all_results)

    return file_name

def fetch_historical_data(securities, fields, start_date, end_date, fill='P', days='W'):
    """
    Fetch historical data using Bloomberg API.
    """
    params = {
        'securities': securities,
        'fields': fields,
        'start_date': start_date,
        'end_date': end_date,
        'Fill': fill,
        'Days': days,
    }
    
    logging.info(f"Parameters used for fetching data: {params}")
    
    try:
        data = blp.bdh(
            tickers=params['securities'],
            flds=params['fields'],
            start_date=params['start_date'],
            end_date=params['end_date'],
            Fill=params['Fill'],
            Days=params['Days']
        )
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None
    
    return data

def calculate_start_date(end_date: str, period: str) -> str:
    """
    Calculate the start date based on the end date and the given period.
    """
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    if 'day' in period:
        days = int(period.split()[0])
        start_date_dt = end_date_dt - timedelta(days=days)
    elif 'week' in period:
        weeks = int(period.split()[0])
        start_date_dt = end_date_dt - timedelta(weeks=weeks)
    elif 'month' in period:
        months = int(period.split()[0])
        start_date_dt = end_date_dt - relativedelta(months=months)
    elif 'year' in period:
        years = int(period.split()[0])
        start_date_dt = end_date_dt - relativedelta(years=years)
    else:
        raise ValueError("Period format not recognized. Use 'days', 'weeks', 'months', or 'years'.")
    
    return start_date_dt.strftime('%Y-%m-%d')

def calculate_historical_return(data):
    """
    Calculate historical return using difference.
    """
    return data.diff()

def calculate_percentage_change(data):
    """
    Calculate historical return using percentage change.
    """
    return data.pct_change()

def process_data(data, sort='A', hide_dates=False, dropNA=True):
    """
    Process data: sort, hide dates, and drop NA values if needed.
    """
    if sort == 'D':
        data = data.sort_index(ascending=False)
    if hide_dates:
        data = data.reset_index(drop=True)
    if dropNA:
        data = data.dropna()
    return data

def calculate_var(group_daily_changes, z_scores):
    """
    Calculate VaR for different confidence levels.
    """
    group_std_dev = group_daily_changes.std()
    return {conf_level: z_score * group_std_dev for conf_level, z_score in z_scores.items()}

def cornish_fisher_adjustment(z_score, skewness, kurtosis):
    """
    Apply Cornish-Fisher adjustment to the Z-score using skewness and kurtosis.
    """
    return (z_score 
            + (1/6) * (z_score**2 - 1) * skewness 
            + (1/24) * (z_score**3 - 3*z_score) * kurtosis 
            - (1/36) * (2*z_score**3 - 5*z_score) * (skewness**2))

def calculate_adjusted_var(group_daily_changes, z_scores):
    """
    Calculate VaR with skewness and kurtosis adjustments.
    """
    group_skewness = skew(group_daily_changes)
    group_kurtosis = kurtosis(group_daily_changes, fisher=True)
    group_std_dev = group_daily_changes.std()

    adjusted_VaR = {}
    for conf_level, z_score in z_scores.items():
        adjusted_z = cornish_fisher_adjustment(z_score, group_skewness, group_kurtosis)
        adjusted_VaR[conf_level] = adjusted_z * group_std_dev

    return adjusted_VaR, group_skewness, group_kurtosis

def calculate_historical_var(returns, confidence_levels):
    """
    Calculate Historical VaR for a given set of returns and a list of confidence levels.
    """
    var_values = {}
    for conf_level in confidence_levels:
        var_value = np.percentile(returns, 100 * (1 - conf_level), method='lower')
        var_values[f"{int(conf_level*100)}%"] = abs(var_value)
    return var_values

def compute_dd_tuw(series, dollars=True):
    # Transform the series into cumulative sum
    cum_series = series.expanding().sum()
    
    df0 = cum_series.to_frame('pnl')
    df0['hwm'] = cum_series.expanding().max()

    # Ensure the index is a datetime index
    df0.index = pd.to_datetime(df0.index)
    
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm', 'min']
    df1.index = pd.to_datetime(df0['hwm'].drop_duplicates(keep='first').index)  # time of hwm
    df1 = df1[df1['hwm'] >= df1['min']]  # hwm followed by a drawdown
    
    if dollars:
        dd = df1['hwm'] - df1['min']
    else:
        dd = 1 - df1['min'] / df1['hwm']
    
    tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, 'D')).values  # in days

    # Calculate TUW for the last HWM to the end of the period
    if not df1.empty:
        last_hwm_index = pd.Timestamp(df1.index[-1])
        end_index = pd.Timestamp(df0.index[-1])
        if last_hwm_index != end_index:
            last_tuw = (end_index - last_hwm_index).days
            tuw = np.append(tuw, last_tuw)
            tuw_index = df1.index[:-1].append(pd.Index([end_index]))
            tuw = pd.Series(tuw, index=tuw_index)
        else:
            tuw = pd.Series(tuw, index=df1.index[:-1])
    else:
        tuw = pd.Series()

    return dd, tuw, df0['hwm'], df1.index

def calculate_cumulative_returns(daily_changes):
    # return (1 + daily_changes).cumprod() - 1
    return daily_changes.expanding().sum()

def perform_linear_regression(dependent_var, independent_vars):
    model = LinearRegression()
    model.fit(independent_vars, dependent_var)
    beta = model.coef_
    alpha = model.intercept_
    r_squared = model.score(independent_vars, dependent_var)
    return beta, alpha, r_squared

def perform_beta_analysis(portfolio_cumulative_returns, benchmarks_combined, z_scores, confidence_level=0.95):
    logging.info("Starting beta analysis")
    
    # Convert the series to a dataframe
    portfolio_cumulative_returns_df = portfolio_cumulative_returns.to_frame(name='Portfolio Cumulative Returns')
    
    # Concatenate the portfolio cumulative returns with the combined benchmarks data
    combined_df = pd.concat([portfolio_cumulative_returns_df, benchmarks_combined], axis=1, join='inner')
    
    # Align the data by dropping NA values
    aligned_data = combined_df.dropna()
    
    # Prepare dependent and independent variables
    y = aligned_data.iloc[:, 0]
    X = aligned_data.iloc[:, 1:]
    
    # Perform linear regression
    beta, alpha, r_squared = perform_linear_regression(y, X)
    logging.info(f"Linear regression results - Beta: {beta}, Alpha: {alpha}, R-squared: {r_squared}")

    # Create a dictionary for original beta values
    beta_dict = {benchmark: beta_value for benchmark, beta_value in zip(X.columns, beta)}

    # Calculate the yhat series (linear regression cumulative series)
    yhat = X.dot(beta) + alpha
    logging.info(f"Yhat series:\n{yhat.head()}")
    
    # Convert yhat to a return series by taking the difference
    yhat_returns = yhat.diff().dropna()
    logging.info(f"Yhat returns:\n{yhat_returns.head()}")
    
    # Calculate VaR for the yhat return series for the specific confidence level
    yhat_var = calculate_var(yhat_returns, z_scores)[f"{int(confidence_level*100)}%"]
    logging.info(f"Yhat VaR (confidence level {confidence_level}): {yhat_var}")

    # Calculate VaR for the original cumulative sum series for the specific confidence level
    original_var = calculate_var(y, z_scores)[f"{int(confidence_level*100)}%"]
    logging.info(f"Original VaR (confidence level {confidence_level}): {original_var}")
    
    # Compute the ratio of VaR values
    var_ratio = yhat_var / original_var
    logging.info(f"VaR ratio: {var_ratio}")
    
    # Adjust beta values by multiplying with the ratio of VaR values
    adjusted_beta = {benchmark: beta_value * var_ratio for benchmark, beta_value in zip(X.columns, beta)}
    logging.info(f"Adjusted beta values: {adjusted_beta}")

    return {
        'beta': beta_dict,
        'adjusted_beta': adjusted_beta,
        'alpha': alpha,
        'r_squared': r_squared,
        'var_ratio': var_ratio
    }

def process_group(group_name, group_info, tickers, actual_changes, field_name, confidence_levels, z_scores, benchmarks_combined):
    group_changes = pd.Series(0, index=actual_changes.index, dtype=float)
    logging.info(f"Processing group: {group_name}")
    for idx, size, _ in group_info:
        ticker = tickers[idx]
        if ticker in actual_changes:
            # Extract the Series from the DataFrame using the dynamic field name
            ticker_changes = actual_changes[ticker]
            group_changes = group_changes.add(ticker_changes * size, fill_value=0)
            logging.info(f"Ticker: {ticker}, Size: {size}, Changes: {ticker_changes.head()}")
    
    # Calculate VaR for different confidence levels using predefined Z-scores
    group_VaR = calculate_var(group_changes, z_scores)

    # Calculate adjusted VaR with skewness and kurtosis adjustments
    adjusted_VaR, group_skewness, group_kurtosis = calculate_adjusted_var(group_changes, z_scores)

    # Calculate Historical VaR for the group
    historical_var_group = calculate_historical_var(group_changes, confidence_levels)

    # Calculate Drawdown (DD) and Time Under Water (TuW)
    group_dd, group_tuw, group_hwm, group_hwm_indices = compute_dd_tuw(group_changes)

    group_cumulative_returns = calculate_cumulative_returns(group_changes)

    # Perform beta analysis for the group
    logging.info(f"Processing Beta Analysis for group: {group_name}")
    beta_analysis_results = perform_beta_analysis(group_cumulative_returns, benchmarks_combined, z_scores)

    return {
        'group_daily_changes': group_changes,
        'group_cumulative_returns': group_cumulative_returns,
        'group_mean': group_changes.mean(),
        'group_std_dev': group_changes.std(),
        'group_VaR': group_VaR,
        'adjusted_VaR': adjusted_VaR,
        'group_skewness': group_skewness,
        'group_kurtosis': group_kurtosis,
        'historical_var_group': historical_var_group,
        'group_dd': group_dd,
        'group_tuw': group_tuw,
        'group_hwm': group_hwm,
        'group_hwm_indices': group_hwm_indices,
        'beta_analysis': beta_analysis_results
    }

def process_portfolio(groups, tickers, actual_changes, field_name, confidence_levels, z_scores, benchmarks_combined):
    portfolio_changes = pd.Series(0, index=actual_changes.index, dtype=float)
    for group_info in groups.values():
        for idx, size, _ in group_info:
            ticker = tickers[idx]
            if ticker in actual_changes:
                ticker_changes = actual_changes[ticker]
                portfolio_changes = portfolio_changes.add(ticker_changes * size, fill_value=0)
    
    # logging.info(f"Portfolio Changes: {portfolio_changes.head()}")
    
    # Calculate VaR for different confidence levels using predefined Z-scores
    portfolio_VaR = calculate_var(portfolio_changes, z_scores)
    
    # Calculate adjusted VaR with skewness and kurtosis adjustments
    adjusted_portfolio_VaR, portfolio_skewness, portfolio_kurtosis = calculate_adjusted_var(portfolio_changes, z_scores)
    
    # Calculate Historical VaR for the portfolio
    historical_var_portfolio = calculate_historical_var(portfolio_changes, confidence_levels)

    # Calculate Drawdown (DD) and Time Under Water (TuW)
    portfolio_dd, portfolio_tuw, portfolio_hwm, portfolio_hwm_indices = compute_dd_tuw(portfolio_changes)

    portfolio_cumulative_returns = calculate_cumulative_returns(portfolio_changes)

    # Perform beta analysis for the portfolio
    logging.info(f"Processing Beta Analysis for Portfolio")
    beta_analysis_results = perform_beta_analysis(portfolio_cumulative_returns, benchmarks_combined, z_scores)

    return {
        'portfolio_changes': portfolio_changes,
        'portfolio_cumulative_returns': portfolio_cumulative_returns,
        'portfolio_mean': portfolio_changes.mean(),
        'portfolio_std_dev': portfolio_changes.std(),
        'portfolio_VaR': portfolio_VaR,
        'adjusted_VaR': adjusted_portfolio_VaR,
        'portfolio_skewness': portfolio_skewness,
        'portfolio_kurtosis': portfolio_kurtosis,
        'historical_var_portfolio': historical_var_portfolio,
        'portfolio_dd': portfolio_dd,
        'portfolio_tuw': portfolio_tuw,
        'portfolio_hwm': portfolio_hwm,  # Include HWM points
        'portfolio_hwm_indices': portfolio_hwm_indices,
        'beta_analysis': beta_analysis_results
    }

def calculate_actual_changes(historical_data, tickers, groups, field_name):
    actual_changes = pd.DataFrame(index=historical_data.index)
    
    for group_name, group_info in groups.items():
        for idx, size, ticker_type in group_info:
            ticker = tickers[idx]
            if ticker in historical_data:
                ticker_data = historical_data[ticker][field_name]
                
                if ticker_type == 'Rates':
                    # Calculate changes in basis points for Rates
                    logging.info(f"Rates Ticker: {ticker}, Rates Ticker Data: {ticker_data.head()}")
                    actual_change = (ticker_data - ticker_data.shift(1)) * 1e5  # Basis point to actual changes
                    logging.info(f"Rates Ticker: {ticker}, Rates Actual Change: {actual_change.head()}")
                else:
                    # Calculate percentage change for other types
                    percentage_change = calculate_percentage_change(ticker_data)
                    processed_percentage_change = process_data(percentage_change, sort='A', dropNA=True)
                    actual_change = processed_percentage_change * 1e6

                actual_changes[ticker] = actual_change

    return actual_changes.dropna()

def fetch_and_combine_benchmarks(start_date, end_date):
    # Fetch benchmark data
    benchmarks = ['DXY Curncy', 'USDCNH Curncy', 'SPX Index', 'CO1 Comdty']
    benchmark_data = fetch_historical_data(benchmarks, ['Last Price'], start_date, end_date)
    benchmarks_log_returns = np.log(benchmark_data)
    
    # Fetch benchmark rates data
    benchmarks_rates = ['USOSFR10 CMPN Curncy', 'USOSFR10 CMPT Curncy']
    benchmarks_rates_data = fetch_historical_data(benchmarks_rates, ['Last Price'], start_date, end_date)
    
    # Concatenate benchmark log returns with benchmark rates data
    benchmarks_combined = pd.concat([benchmarks_log_returns, benchmarks_rates_data], axis=1, join='inner')
    
    return benchmarks_combined

def calculate_var_artifacts(tickers, fields, period, end_date, confidence_levels, z_scores, groups=None):
    start_date = calculate_start_date(end_date, period)
    historical_data = fetch_historical_data(tickers, fields, start_date, end_date)
    
    if (historical_data) is None:
        return

    field_name = fields[0]  # Assuming there's only one field in the list

    # Calculate actual changes using the new method
    actual_changes = calculate_actual_changes(historical_data, tickers, groups, field_name)

    # Fetch and combine benchmarks data
    benchmarks_combined = fetch_and_combine_benchmarks(start_date, end_date)
    
    group_results = {}

    # Calculate for each group
    for group_name, group_info in groups.items():
        group_results[group_name] = process_group(group_name, group_info, tickers, actual_changes, field_name, confidence_levels, z_scores, benchmarks_combined)

    # Calculate portfolio results
    portfolio_results = process_portfolio(groups, tickers, actual_changes, field_name, confidence_levels, z_scores, benchmarks_combined)

    # Create a dictionary with all artifacts
    results = {
        'start_date': start_date,
        'end_date': end_date,
        'historical_data': historical_data,
        # 'percentage_changes': percentage_changes,
        # 'processed_percentage_changes': processed_percentage_changes,
        'actual_changes': actual_changes,
        'group_results': group_results,
        'portfolio_results': portfolio_results
    }

    return results

def main(tickers, fields, periods, end_date, confidence_levels, z_scores, groups=None):
    all_results = {}
    for period in periods:
        results = calculate_var_artifacts(tickers, fields, period, end_date, confidence_levels, z_scores, groups)
        if results:
            all_results[period] = results
            logging.info(f"Results for period {period} have been calculated and saved.")

            # Log the beta analysis results for each group
            for group_name, group_result in results['group_results'].items():
                beta_analysis_results = group_result['beta_analysis']
                logging.info(f"Beta analysis results for group {group_name} in period {period}: {beta_analysis_results['beta']}")
            
            # Log the beta analysis results for the portfolio
            beta_analysis_results = results['portfolio_results']['beta_analysis']
            logging.info(f"Beta analysis results for portfolio in period {period}: {beta_analysis_results['beta']}")
    
    file_name = save_results_to_file(all_results, groups, tickers, fields, confidence_levels)  # Pass the required arguments
    return all_results, file_name