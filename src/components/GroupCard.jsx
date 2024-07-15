// src/components/GroupCard.jsx
import React from 'react';
import '../styles/GroupCard.css';

function GroupCard({ group, index, updateGroup, removeGroup, tickers }) {
  const addItem = () => {
    const newItem = { tickerIndex: 0, size: '', type: 'FX' };
    const updatedGroup = { ...group, items: [...group.items, newItem] };
    updateGroup(index, updatedGroup);
  };

  const updateItem = (itemIndex, key, value) => {
    const updatedItems = group.items.map((item, i) =>
      i === itemIndex ? { ...item, [key]: value } : item
    );
    updateGroup(index, { ...group, items: updatedItems });
  };

  const removeItem = (itemIndex) => {
    const updatedItems = group.items.filter((_, i) => i !== itemIndex);
    updateGroup(index, { ...group, items: updatedItems });
  };

  const updateGroupName = (name) => {
    updateGroup(index, { ...group, name });
  };

  return (
    <div className="group-card">
      <input
        type="text"
        value={group.name}
        onChange={(e) => updateGroupName(e.target.value)}
        className="group-name"
      />
      {group.items.map((item, itemIndex) => (
        <div key={itemIndex} className="group-item">
          <select
            value={item.tickerIndex}
            onChange={(e) => updateItem(itemIndex, 'tickerIndex', e.target.value)}
            className="item-select"
          >
            {tickers.map((ticker, idx) => (
              <option key={idx} value={idx}>{ticker}</option>
            ))}
          </select>
          <input
            type="number"
            value={item.size}
            onChange={(e) => updateItem(itemIndex, 'size', e.target.value)}
            placeholder="Size"
            className="item-input"
          />
          <select
            value={item.type}
            onChange={(e) => updateItem(itemIndex, 'type', e.target.value)}
            className="item-select"
          >
            <option value="FX">FX</option>
            <option value="CDS">CDS</option>
            <option value="Rates">Rates</option>
            <option value="Equity (ETF, Index, Futures)">Equity (ETF, Index, Futures)</option>
            <option value="Credit Bonds">Credit Bonds</option>
            <option value="Sov Bonds">Sov Bonds</option>
          </select>
          <button
            onClick={() => removeItem(itemIndex)}
            className="remove-button"
          >
            Remove
          </button>
        </div>
      ))}
      <button
        onClick={addItem}
        className="add-item-button"
      >
        Add Item
      </button>
      <button
        onClick={() => removeGroup(index)}
        className="remove-group-button"
      >
        Remove Group
      </button>
    </div>
  );
}

export default GroupCard;