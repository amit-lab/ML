import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('USA_Housing.csv')

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

# Split the x and y data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

# importing linear regression model and fit the test data
lm = LinearRegression()
lm.fit(x_train, y_train)

cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])

predictions = lm.predict(x_test)
plt.scatter(y_test, predictions)

plt.show()
