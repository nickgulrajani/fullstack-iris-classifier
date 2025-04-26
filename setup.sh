#!/bin/bash

echo "Installing backend dependencies..."
cd backend
npm install
pip install -r requirements.txt
python3 train_model.py

echo "Installing frontend dependencies..."
cd ../frontend
npm install

echo "✅ All dependencies and model generated!"

