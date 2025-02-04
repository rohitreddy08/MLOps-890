import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv(r"C:\Users\RRMSU\Downloads\sampregdata.csv")

# Two predictors (x2, x4)
X_two = data[['x2', 'x4']]
y = data['y']

model_two = LinearRegression()
model_two.fit(X_two, y)

# Print performance
print("Two Predictor Model (x2 and x4)")
print("Intercept:", model_two.intercept_)
print("Coefficients (x2, x4):", model_two.coef_)
print("R^2 score:", model_two.score(X_two, y))
