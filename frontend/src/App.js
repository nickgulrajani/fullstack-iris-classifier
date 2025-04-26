import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [features, setFeatures] = useState({ sepalLength: '', sepalWidth: '', petalLength: '', petalWidth: '' });
  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setFeatures({
      ...features,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const featureArray = [
      parseFloat(features.sepalLength),
      parseFloat(features.sepalWidth),
      parseFloat(features.petalLength),
      parseFloat(features.petalWidth),
    ];

    try {
      const response = await axios.post('http://localhost:3001/predict', { features: featureArray });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error(error);
      alert('Prediction failed!');
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Iris Flower Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="number"
          name="sepalLength"
          placeholder="Sepal Length"
          value={features.sepalLength}
          onChange={handleChange}
          required
        /><br /><br />
        <input
          type="number"
          name="sepalWidth"
          placeholder="Sepal Width"
          value={features.sepalWidth}
          onChange={handleChange}
          required
        /><br /><br />
        <input
          type="number"
          name="petalLength"
          placeholder="Petal Length"
          value={features.petalLength}
          onChange={handleChange}
          required
        /><br /><br />
        <input
          type="number"
          name="petalWidth"
          placeholder="Petal Width"
          value={features.petalWidth}
          onChange={handleChange}
          required
        /><br /><br />
        <button type="submit">Predict</button>
      </form>

      {prediction !== null && (
        <div style={{ marginTop: '20px' }}>
          <h2>Prediction: {renderFlower(prediction)}</h2>
        </div>
      )}
    </div>
  );
}

function renderFlower(prediction) {
  const pred = parseInt(prediction);  // 👈 Corrected: Parse prediction as integer
  switch (pred) {
    case 0:
      return 'Iris Setosa';
    case 1:
      return 'Iris Versicolour';
    case 2:
      return 'Iris Virginica';
    default:
      return 'Unknown';
  }
}

export default App;

