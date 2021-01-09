from joblib import dump, load
import numpy as np
model = load('Dragon.joblib') 

features = np.array([]])
       
result=model.predict(features)

print(result)

