// src/App.jsx
import React, { useState } from 'react';
import TickerInput from './components/TickerInput';
import GroupCard from './components/GroupCard';
import ParametersInput from './components/ParametersInput';
import './styles/App.css';

function App() {
  const [tickers, setTickers] = useState([]);
  const [groups, setGroups] = useState([]);
  const [parameters, setParameters] = useState({ periods: [], endDate: '' });
  const [errorMessage, setErrorMessage] = useState('');

  const addGroup = () => {
    const newGroup = { name: `Group ${groups.length + 1}`, items: [] };
    setGroups([...groups, newGroup]);
  };

  const updateGroup = (index, updatedGroup) => {
    const updatedGroups = groups.map((group, i) => (i === index ? updatedGroup : group));
    setGroups(updatedGroups);
  };

  const removeGroup = (index) => {
    const updatedGroups = groups.filter((_, i) => i !== index);
    setGroups(updatedGroups);
  };

  const validateInputs = () => {
    for (const group of groups) {
      if (!group.name.trim()) {
        setErrorMessage('All group names must be filled.');
        return false;
      }
      for (const item of group.items) {
        if (item.size === '' || item.type === '' || item.tickerIndex === '') {
          setErrorMessage('All group items must be filled.');
          return false;
        }
      }
    }
    if (parameters.periods.length === 0 || !parameters.endDate) {
      setErrorMessage('Please specify periods and end date.');
      return false;
    }
    setErrorMessage('');
    return true;
  };

  const generateOutput = () => {
    if (validateInputs()) {
      // Process tickers
      const processedTickers = tickers;
  
      // Process groups
      const processedGroups = {};
      groups.forEach((group) => {
        processedGroups[group.name] = group.items.map((item) => [
          processedTickers.indexOf(item.ticker),
          parseFloat(item.size),
          item.type,
        ]);
      });
  
      // Extract periods and end date from parameters
      const { periods, endDate } = parameters;
  
      // Log the processed inputs
      console.log('Tickers:', processedTickers);
      console.log('Groups:', processedGroups);
      console.log('Periods:', periods);
      console.log('End Date:', endDate);
  
      // Define the fixed values
      const fields = ['Last Price'];
      const confidence_levels = [0.90, 0.95, 0.99];
      const z_scores = { '90%': 1.28, '95%': 1.65, '99%': 2.33 };
  
      // Call the main function (this would be integrated with your backend)
      // results = main(processedTickers, fields, periods, endDate, confidence_levels, z_scores, processedGroups);
      // if (results) {
      //   console.log("All artifacts and data have been processed and displayed.");
      // }
    }
  };

  return (
    <div className="container">
      <h1 className="title">Portfolio Manager</h1>
      <TickerInput tickers={tickers} setTickers={setTickers} />
      <ParametersInput parameters={parameters} setParameters={setParameters} />
      <div className="button-container">
        <button
          onClick={addGroup}
          className="add-group-button"
        >
          Add Group
        </button>
      </div>
      {groups.map((group, index) => (
        <GroupCard
          key={index}
          group={group}
          index={index}
          updateGroup={updateGroup}
          removeGroup={removeGroup}
          tickers={tickers}
        />
      ))}
      {errorMessage && (
        <div className="error-message">
          <span>{errorMessage}</span>
        </div>
      )}
      <button
        onClick={generateOutput}
        className="generate-button"
      >
        Generate
      </button>
    </div>
  );
}

export default App;