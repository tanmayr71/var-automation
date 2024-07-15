// src/components/GroupInput.jsx
import React, { useState } from 'react';

const GroupInput = ({ tickers, addGroup, groups, removeGroup, updateGroup }) => {
  const [groupName, setGroupName] = useState('');
  const [groupItems, setGroupItems] = useState([]);

  const addItem = () => {
    setGroupItems([...groupItems, { ticker: '', weight: '', type: 'FX' }]);
  };

  const handleItemChange = (index, field, value) => {
    const newItems = groupItems.map((item, i) => (i === index ? { ...item, [field]: value } : item));
    setGroupItems(newItems);
  };

  const handleAddGroup = () => {
    if (groupName && groupItems.length > 0) {
      addGroup(groupName, groupItems);
      setGroupName('');
      setGroupItems([]);
    }
  };

  const handleRemoveItem = (groupIndex, itemIndex) => {
    const updatedGroups = groups.map((group, index) => {
      if (index === groupIndex) {
        const newItems = group.items.filter((_, i) => i !== itemIndex);
        return { ...group, items: newItems };
      }
      return group;
    });
    updateGroup(updatedGroups);
  };

  return (
    <div className="mb-4">
      <div className="flex space-x-2 mb-2">
        <input
          type="text"
          value={groupName}
          onChange={(e) => setGroupName(e.target.value)}
          placeholder="Enter group name"
          className="border p-2 rounded w-full"
        />
        <button onClick={handleAddGroup} className="bg-blue-500 text-white px-4 py-2 rounded">
          Add Group
        </button>
      </div>
      {groups.map((group, groupIndex) => (
        <div key={groupIndex} className="border p-4 rounded mb-4 bg-gray-100">
          <div className="flex justify-between items-center mb-2">
            <strong>{group.name}</strong>
            <button onClick={() => removeGroup(groupIndex)} className="text-red-500">Remove Group</button>
          </div>
          {group.items.map((item, itemIndex) => (
            <div key={itemIndex} className="flex space-x-2 mb-2">
              <select
                value={item.ticker}
                onChange={(e) => handleItemChange(groupIndex, itemIndex, 'ticker', e.target.value)}
                className="border p-2 rounded w-full"
              >
                <option value="">Select ticker</option>
                {tickers.map((ticker, i) => (
                  <option key={i} value={ticker}>
                    {ticker}
                  </option>
                ))}
              </select>
              <input
                type="number"
                value={item.weight}
                onChange={(e) => handleItemChange(groupIndex, itemIndex, 'weight', e.target.value)}
                placeholder="Weight"
                className="border p-2 rounded w-full"
              />
              <select
                value={item.type}
                onChange={(e) => handleItemChange(groupIndex, itemIndex, 'type', e.target.value)}
                className="border p-2 rounded w-full"
              >
                <option value="FX">FX</option>
                <option value="CDS">CDS</option>
                <option value="Rates">Rates</option>
                <option value="Equity">Equity</option>
              </select>
              <button onClick={() => handleRemoveItem(groupIndex, itemIndex)} className="text-red-500">x</button>
            </div>
          ))}
          <button onClick={addItem} className="bg-green-500 text-white px-4 py-2 rounded">
            Add Item
          </button>
        </div>
      ))}
    </div>
  );
};

export default GroupInput;