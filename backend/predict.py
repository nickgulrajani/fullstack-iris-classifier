import sys
import joblib
import numpy as np

model = joblib.load('model.pkl')
input_features = list(map(float, sys.argv[1:]))
input_array = np.array(input_features).reshape(1, -1)
prediction = model.predict(input_array)
print(int(prediction[0]))