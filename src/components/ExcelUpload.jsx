import React, { useRef, useState } from 'react';
import axios from 'axios';
import * as XLSX from 'xlsx';
import Spinner from './Spinner';
import '../styles/ExcelUpload.css';

const ExcelUpload = ({ setTickers, setGroups }) => {
  const fileInputRef = useRef(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [loading, setLoading] = useState(false); // Add loading state

  // Get the backend URL from environment variables
  const backendUrl = import.meta.env.VITE_BACKEND_URL;

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setLoading(true);
      const reader = new FileReader();
      reader.onload = (e) => {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const json = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

        // Log the extracted rows for debugging
        console.log('Excel data:', json);

        // Extract types, sizes, exact tickers, and group information
        const types = json[0].slice(1); // 3rd row
        const sizes = json[1].slice(1); // 4th row
        const tickers = json[2].slice(1); // 5th row
        const group_ids = json[3].slice(1); // 6th row

        // Combine the extracted rows into a DataFrame-like structure for easier processing
        const combinedData = tickers.map((ticker, index) => ({
          type: types[index],
          size: sizes[index],
          ticker: ticker,
          group_id: group_ids[index],
        }));

        // Filter out rows with any empty or NaN values
        const filteredData = combinedData.filter(item => item.type && item.size && item.ticker && item.group_id);

        // Process tickers
        const processedTickers = filteredData.map(item => item.ticker);

        // Validate tickers before proceeding
        axios.post(`${backendUrl}/api/validate_tickers`, {
          tickers: processedTickers,
        })
        .then(response => {
          const { valid_tickers, invalid_tickers } = response.data;
          if (invalid_tickers.length > 0) {
            setErrorMessage(`Invalid tickers: ${invalid_tickers.join(', ')}`);
            setLoading(false);
            return;
          }

          // Clear any previous error message
          setErrorMessage('');

          // Process groups
          const processedGroups = {};
          filteredData.forEach((item, index) => {
            const group_name = 'Group ' + item.group_id;
            const tickerIndex = processedTickers.indexOf(item.ticker);
            if (processedGroups[group_name]) {
              processedGroups[group_name].push({
                ticker: item.ticker,
                tickerIndex: tickerIndex,
                size: item.size,
                type: item.type,
              });
            } else {
              processedGroups[group_name] = [{
                ticker: item.ticker,
                tickerIndex: tickerIndex,
                size: item.size,
                type: item.type,
              }];
            }
          });

          // Update the state
          setTickers(processedTickers);
          setGroups(Object.entries(processedGroups).map(([name, items]) => ({ name, items })));

          // Log the processed data for debugging
          console.log('Processed Tickers:', processedTickers);
          console.log('Processed Groups:', processedGroups);
          setLoading(false);
        })
        .catch(error => {
          console.error('There was an error validating the tickers!', error);
          setErrorMessage('Error validating tickers');
          setLoading(false); // Set loading to false if an error occurs
        });
      };
      reader.readAsArrayBuffer(file);
    }
  };

  return (
    <div className="excel-upload">
      <input
        type="file"
        accept=".xlsx, .xls"
        ref={fileInputRef}
        onChange={handleFileUpload}
        style={{ display: 'none' }}
      />
      <button onClick={() => fileInputRef.current.click()} className="upload-button">
        Upload Excel
      </button>
      {loading && <Spinner />} {/* Render Spinner when loading */}
      {errorMessage && <div className="error-message">{errorMessage}</div>}
    </div>
  );
};

export default ExcelUpload;