import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"C:\Users\RRMSU\Downloads\sampregdata.csv")

corr = data.corr()
print("\nCorrelation matrix:")
print(corr)

# Single predictor (x4)
X_single = data[['x4']]   
y = data['y']

model_single = LinearRegression()
model_single.fit(X_single, y)

# Print performance and plot
print("Single Predictor Model (x4)")
print("Intercept:", model_single.intercept_)
print("Coefficient for x4:", model_single.coef_[0])
print("R^2 score:", model_single.score(X_single, y))

plt.scatter(X_single, y, color='blue', label='Data points')
x_range = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
plt.plot(x_range, model_single.predict(x_range), color='red', label='Regression line')
plt.xlabel('x4')
plt.ylabel('y')
plt.legend()
plt.title('Single Predictor Model (x4)')
plt.show()
