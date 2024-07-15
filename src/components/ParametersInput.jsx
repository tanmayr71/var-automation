// src/components/ParametersInput.jsx
import React, { useState, useEffect } from 'react';
import '../styles/ParametersInput.css';

const ParametersInput = ({ parameters, setParameters }) => {
  const [period, setPeriod] = useState('');
  const [endDate, setEndDate] = useState(parameters.endDate || '');

  useEffect(() => {
    setParameters({ periods: parameters.periods, endDate });
  }, [endDate]);

  const addPeriod = () => {
    if (period.trim() !== '') {
      setParameters({
        periods: [...parameters.periods, period.trim()],
        endDate,
      });
      setPeriod('');
    }
  };

  const removePeriod = (index) => {
    const newPeriods = parameters.periods.filter((_, i) => i !== index);
    setParameters({
      periods: newPeriods,
      endDate,
    });
  };

  return (
    <div className="parameters-input">
      <h2 className="parameters-title">Parameters</h2>
      <div className="parameters-item">
        <input
          type="text"
          value={period}
          onChange={(e) => setPeriod(e.target.value)}
          placeholder="Enter Period (e.g., 100 days, 3 weeks, 6 months, 1 year)"
          className="parameters-input-field"
        />
        <button
          onClick={addPeriod}
          className="parameters-button"
        >
          Add Period
        </button>
      </div>
      <div>
        {parameters.periods.map((period, index) => (
          <div key={index} className="parameters-item">
            <span className="parameters-period">{period}</span>
            <button
              onClick={() => removePeriod(index)}
              className="parameters-remove-button"
            >
              Remove
            </button>
          </div>
        ))}
      </div>
      <div className="parameters-item">
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          className="parameters-input-field"
        />
      </div>
    </div>
  );
};

export default ParametersInput;