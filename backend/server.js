const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

app.post('/predict', (req, res) => {
    const { features } = req.body;
    if (!features || !Array.isArray(features)) {
        return res.status(400).json({ error: 'Features must be provided as an array.' });
    }
    const command = `python3 predict.py ${features.join(' ')}`;
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error.message}`);
            return res.status(500).json({ error: 'Prediction failed.' });
        }
        const prediction = stdout.trim();
        res.json({ prediction });
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});