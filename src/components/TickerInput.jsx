import React, { useState } from 'react';
import axios from 'axios';
import '../styles/TickerInput.css';

const TickerInput = ({ tickers, setTickers }) => {
  const [ticker, setTicker] = useState('');
  const [collapsed, setCollapsed] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Get the backend URL from environment variables
  const backendUrl = import.meta.env.VITE_BACKEND_URL;

  const addTicker = () => {
    if (ticker.trim() === '') {
      setErrorMessage('Ticker cannot be empty');
      return;
    }

    // Call the validation API
    axios.post(`${backendUrl}/api/validate_tickers`, {
      tickers: [ticker.trim()],
    })
    .then(response => {
      const { valid_tickers, invalid_tickers } = response.data;
      if (valid_tickers.length > 0) {
        setTickers([...tickers, ticker.trim()]);
        setTicker('');
        setErrorMessage('');
      } else {
        setErrorMessage('Invalid ticker: ' + invalid_tickers[0]);
      }
    })
    .catch(error => {
      console.error('There was an error validating the ticker!', error);
      setErrorMessage('Error validating ticker');
    });
  };

  const removeTicker = (index) => {
    const newTickers = tickers.filter((_, i) => i !== index);
    setTickers(newTickers);
  };

  const toggleCollapse = () => {
    setCollapsed(!collapsed);
  };

  return (
    <div className="ticker-input">
      <div className="ticker-title">
        <span>Tickers</span>
        <button
          onClick={toggleCollapse}
          className="ticker-collapse-button"
        >
          {collapsed ? 'Expand' : 'Collapse'}
        </button>
      </div>
      {!collapsed && (
        <>
          <div className="ticker-item">
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="Enter Ticker"
              className="ticker-input-field"
            />
            <button
              onClick={addTicker}
              className="ticker-add-button"
            >
              Add
            </button>
          </div>
          {errorMessage && <div className="error-message">{errorMessage}</div>}
          <div className="ticker-list">
            {tickers.map((ticker, index) => (
              <div key={index} className="ticker-item">
                <span className="ticker-name">{ticker}</span>
                <button
                  onClick={() => removeTicker(index)}
                  className="ticker-remove-button"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default TickerInput;
