Project Title: Employee Salary Prediction

📝Overview
This project focuses on building a machine learning model that can predict the salary of
employees based on various input features. It uses regression algorithms—primarily Linear Regression—to estimate the salary,
using a dataset that includes employee details like years of experience.

The notebook walks through all essential stages of a data science project:

Data Loading

1.Exploratory Data Analysis (EDA)
2.Data Preprocessing
3.Model Training
4.Model Evaluation
5.Prediction

📂 Notebook Breakdown and Explanation
1.  Importing Libraries
 
The first step is importing necessary Python libraries:
-import pandas as pd
-import numpy as np
-import matplotlib.pyplot as plt
-import seaborn as sns
-from sklearn.linear_model import LinearRegression
-from sklearn.model_selection import train_test_split
-from sklearn.metrics import r2_score, mean_squared_error

These libraries are used for:

1.Data handling (pandas, numpy)
2.Data visualization (matplotlib, seaborn)
3.Machine Learning modeling and evaluation (scikit-learn)

2. 📁 Data Loading

 data = pd.read_csv("Employee.csv")
 Load the dataset into a DataFrame

4. 🔍 Exploratory Data Analysis (EDA)

- data.info()
-data.describe()
-data.isnull().sum()

To understand the structure of the dataset:
-Check for missing value
- data types
-View basic statistics

4. 🔧 Data Preprocessing

   Before training, we split the dataset:
   X = data[['YearsExperience']]
   y = data['Salary']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    This splits the data into training (80%) and testing (20%) sets.

6. 🤖 Model Training

    model = LinearRegression()
    model.fit(X_train, y_train)

   We fit a Linear Regression model using training data. It tries to find the best-fitting straight line between experience and salary.

8. 📈 Prediction and Evaluation

    y_pred = model.predict(X_test)
 
   -Model evaluation metrics:
   r2_score(y_test, y_pred)
   mean_squared_error(y_test, y_pred)

- These evaluate how well the model is performing:
R² score shows how much of the variation in salary is explained by experience.
MSE gives the average squared difference between predicted and actual salaries.

7. 🖼️ Results Visualization

    plt.scatter(X_test, y_test, color='red')
   plt.plot(X_test, y_pred, color='blue')
   plt.title('Salary vs Experience (Test Set)')
   plt.xlabel('Years of Experience')
   plt.ylabel('Salary')
   plt.show()

-This shows a visual comparison between predicted and actual salaries on the test set.


