// src/components/TickerInput.jsx
import React, { useState } from 'react';
import '../styles/TickerInput.css';

const TickerInput = ({ tickers, setTickers }) => {
  const [ticker, setTicker] = useState('');
  const [collapsed, setCollapsed] = useState(false);

  const addTicker = () => {
    if (ticker.trim() !== '') {
      setTickers([...tickers, ticker.trim()]);
      setTicker('');
    }
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