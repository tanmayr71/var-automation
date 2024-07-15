import React, { useState } from 'react';
import TickerInput from './components/TickerInput';
import GroupCard from './components/GroupCard';
import ParametersInput from './components/ParametersInput';
import ExcelUpload from './components/ExcelUpload';
import Spinner from './components/Spinner';
import './styles/App.css';
import axios from 'axios';

function App() {
  const [tickers, setTickers] = useState([]);
  const [groups, setGroups] = useState([]);
  const [parameters, setParameters] = useState({ periods: [], endDate: '' });
  const [errorMessage, setErrorMessage] = useState('');
  const [loading, setLoading] = useState(false);

  // Get the backend URL from environment variables
  const backendUrl = import.meta.env.VITE_BACKEND_URL;

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
      setLoading(true);
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

      // Send data to backend
      console.log('backendUrl:', backendUrl);
      axios.post(`${backendUrl}/api/save_parameters`, {
        tickers: processedTickers,
        periods,
        endDate,
        groups: processedGroups,
      })
      .then(response => {
        console.log('Data successfully sent to backend:', response.data);

        // Download the generated file
        const resultsPath = response.data.results_path;
        const link = document.createElement('a');
        link.href = `${backendUrl}/api/download?file_path=${encodeURIComponent(resultsPath)}`;
        link.setAttribute('download', 'results.xlsx');
        document.body.appendChild(link);
        link.click();
        link.parentNode.removeChild(link);

        setLoading(false); // Set loading to false
      })
      .catch(error => {
        console.error('There was an error!', error);
        setLoading(false);
      });
    }
  };

  return (
    <div className="container">
      <h1 className="title">Portfolio Manager</h1>
      <ExcelUpload setTickers={setTickers} setGroups={setGroups} />
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
      {loading ? (
        <Spinner />
      ) : (
        <button
          onClick={generateOutput}
          className="generate-button"
          disabled={loading} // Disable the button when loading
        >
          Generate
        </button>
      )}
    </div>
  );
}

export default App;