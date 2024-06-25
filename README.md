NotALoanSharkML_Assessment_Ariff_Hakimi

Overview flow of programs:

Step 1: labelencoder.py -> Data cleaned up,
Step 2: simplestats.py -> Basic statistic,
Step 3: eda.py -> EDA task,
Step 4: combinemodel.py -> Model training, building and testing

==================================================
# Step 1)  Data cleaning 
==================================================

Original dataset: loan_default_data.csv

1) Removing some unrelevant row (first row) and columns ( issue_date,zip_code,address,earlist_creadit_line,last_payment_amnt,last_payment_date,next_payment,last_credit_pull_date)
   Reason: Irrelevant dataset could affect the machine learning model process. To know whether the column is relevant or not, at this point of assessment, I have to 
   consider with my logic. In the real world, there ll be few stakeholders that should decided on the data relevancy.

2) Convert non numerical column values to numerical by using labelencoder method - program name to do conversion: labelencoder.py.
   Reason: First reason is machine learning model trained better with numerical data compare with mixture of numerical and non numerical. Second reason will be,
   this conversion is done initially instead of together during EDA is I want to avoid the python program that run EDA task get overloaded. This conversion task
   itself already take some time to process.
    2.1) This will create a new file name encoded_dataset.csv
    2.2) Encoded columns : 'term', 'employment_length', 'home_ownership','verification_status','loan_status','purpose'
         Reason of doing this:  
    2.3) Mapping for term:
            36 months --> 0
            60 months --> 1

            Mapping for employment_length:
            1 year --> 0
            10+ years --> 1
            2 years --> 2
            3 years --> 3
            4 years --> 4
            5 years --> 5
            6 years --> 6
            7 years --> 7
            8 years --> 8
            9 years --> 9
            < 1 year --> 10
            nan --> 11

            Mapping for home_ownership:
            MORTGAGE --> 0
            NONE --> 1
            OTHER --> 2
            OWN --> 3
            RENT --> 4

            Mapping for verification_status:
            Not Verified --> 0
            Source Verified --> 1
            Verified --> 2

            Mapping for loan_status:
            Charged Off --> 0
            Current --> 1
            Default --> 2
            Does not meet the credit policy. Status:Charged Off --> 3
            Does not meet the credit policy. Status:Fully Paid --> 4
            Fully Paid --> 5
            In Grace Period --> 6
            Late (16-30 days) --> 7
            Late (31-120 days) --> 8

            Mapping for purpose:
            car --> 0
            credit_card --> 1
            debt_consolidation --> 2
            educational --> 3
            home_improvement --> 4
            house --> 5
            major_purchase --> 6
            medical --> 7
            moving --> 8
            other --> 9
            renewable_energy --> 10
            small_business --> 11
            vacation --> 12
            wedding --> 13

        2.4) Changing percentage value to decimal

Hence, the cleaned dataset to be used for training and testing is encoded_dataset.csv.

<pre>
```labelencoder.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('loan_default_data_csv.csv',encoding='utf-8')

# List of columns to encod
columns_to_encode = ['term', 'employment_length', 'home_ownership','verification_status','loan_status','purpose']

# Initialize LabelEncoder
label_encoders = {}

# Iterate over each column to encode
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    df[column + '_encoded'] = label_encoders[column].fit_transform(df[column])

for column, encoder in label_encoders.items():
    print(f"Mapping for {column}:")
    for label, encoded_value in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        print(f"{label} --> {encoded_value}")
    print()


df.to_csv('encoded_dataset.csv', index=False)

print(df.head())
```
</pre>

==================================================
# Step 2) Basic statistic and analysis on the data
==================================================
1) To know on the count, mean,std, min, 25% percentile,50% percentile, 75% percentile
& max count : program to run this task - simplestats.py

<pre>
```simplestats.py
import pandas as pd

data = pd.read_csv('encoded_dataset.csv',encoding='utf-8')
statistics = data.describe(include='all')

print(statistics)
```
</pre>


==================================================
# Step 3) Perform EDA - Exploratory Data Analysis
==================================================

1) Program that performing EDA task: eda.py

2) In this EDA, there are a few things have been achieved:

    2.1) Univariate Analysis:
            -Target Variable Analysis: Analyze the distribution of the target variable.
            -Feature Analysis: Analyze the distribution of each feature (mean, median, mode, standard deviation, histograms, box plots).

    2.2) Bivariate Analysis:

            -Target vs Feature: Analyze the relationship between the target variable and each feature (scatter plots, box plots, bar plots for categorical variables, correlation for numerical variables).

    2.3) Multivariate Analysis:

            -Feature Relationships: Analyze relationships between multiple features (pair plots, correlation matrix, heatmaps).

    2.4) Feature Engineering:

            -Create New Features: Based on domain knowledge and initial insights.
            -Transform Features: Scaling, normalization, encoding categorical variables.

3) Reminder, if you run this eda.py program, you have to go throug all diagrams because I prepare the logic 
   to show box plot, histograms, Scatter plots, Correlation matrix for columns in the dataset - Pair Plot (selected first 5 columns). Just click close icon
   to navigate to next diagrams.

<pre>
```eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('encoded_dataset.csv')

# Basic info on the data
print("First few rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nSummary statistics of the dataset:")
print(df.describe())

print("\nBox plots for numerical features to detect outliers:")
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[column])
    plt.title(f'Box plot of {column}')
    plt.show()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

# Univariate Analysis
print("\nHistograms for numerical features:")
for column in numerical_features:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.show()

# Target variable analysis
target_variable = 'repay_fail'
print(f"\nDistribution of the target variable '{target_variable}':")
plt.figure(figsize=(10, 5))
sns.histplot(df[target_variable], kde=True)
plt.title(f'Distribution of {target_variable}')
plt.show()

# Bivariate Analysis
print("\nScatter plots between target variable and features:")
for column in numerical_features:
    if column != target_variable:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=df[column], y=df[target_variable])
        plt.title(f'Scatter plot between {column} and {target_variable}')
        plt.show()

# Correlation between numerical features
print("\nCorrelation matrix of numerical features:")
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Multivariate Analysis
print("\nPair plot for selected features:")
selected_features = numerical_features[:5]  
sns.pairplot(df[selected_features])
plt.show()

# Feature Engineering
if 'existing_feature1' in df.columns and 'existing_feature2' in df.columns:
    df['new_feature'] = df['existing_feature1'] / df['existing_feature2']

# Example: Encoding categorical variables
df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

# Example: Scaling features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(target_variable, axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns.drop(target_variable))
df_scaled[target_variable] = df[target_variable]

print("\nFirst few rows of the scaled dataset:")
print(df_scaled.head())

```
</pre>



==================================================
# Step 4) Model training and key metrics
==================================================

1) I have selected 3 models to carry out the task which are:
    1.1) Decision Tree Regressor
    1.2) Random Forest Regressor
    1.3) Linear Regression

Reason: The given dataset is having a target variable, which is repay_fail. This means, there will be other features columns that will be used
to do the prediction in resulting the target variable. Hence, generally and commonly, there will be 3 common models that suits to this requirement which are
the one I mentioned above.

2) Program to do the training/building the model: combinemodel.py

3) Evaluation on key metrics for each model:
    3.1) There will be 2 selected key metrics to be used when evaluating the performance of these 3 models:
            3.1.1) Mean Squared Error (MSE)
            3.1.2) R-squared (R²)
    
    Reasons of selection: Apart from commonly used in prediction model, the other main reason I choose these 2 are
    the sensitivity to error and comparison to baseline. To explain more, the dataset given is having a lot of variance in determining
    the target variable, this type of data need to have high sensitivity to error in order to give accurate prediction.

4) Which model is performed better:
To know which is better, we rely on the 2 main metrics. MSE that having lower values are better and 
R2 that is having higher values are better

In the combinemodel.py, I did print some result when I tested with a row of new data.
Based on the result, we do the comparison where the result of the run is:

Decision Tree Regression Performance:
Training set Mean Squared Error (MSE): 0.0
Test set Mean Squared Error (MSE): 0.00012993762993762994
Training set R-squared (R2): 1.0
Test set R-squared (R2): 0.9989969845675597

Random Forest Regression Performance:
Training set Mean Squared Error (MSE): 2.3259591332878537e-06
Test set Mean Squared Error (MSE): 4.142411642411643e-05
Training set R-squared (R2): 0.9999818655254802
Test set R-squared (R2): 0.999680238680138

Linear Regression Performance:
Training set Mean Squared Error (MSE): 0.01866071568078792
Test set Mean Squared Error (MSE): 0.018587089121568152
Training set R-squared (R2): 0.8545106540383076
Test set R-squared (R2): 0.8565224158542498
 
Tested with new data:

Predicted_repay_fail_RF : 0.0
Predicted_repay_fail_RF: 0.0 
Predicted_repay_fail_LR: 0.2

Thus, we can conclude that Random Forest Regression perform better compare to the other 2 because
is has the lowest test MSE and highest test of R2.

<pre>
```combinemodel.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


df = pd.read_csv('encoded_dataset.csv')

# Data separatio
y = df['repay_fail']
X = df.drop('repay_fail', axis=1)

# Splitting dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

y_train_pred_dt = dt_regressor.predict(X_train)
y_test_pred_dt = dt_regressor.predict(X_test)

train_mse_dt = mean_squared_error(y_train, y_train_pred_dt)
test_mse_dt = mean_squared_error(y_test, y_test_pred_dt)
train_r2_dt = r2_score(y_train, y_train_pred_dt)
test_r2_dt = r2_score(y_test, y_test_pred_dt)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_train_pred_rf = rf_regressor.predict(X_train)
y_test_pred_rf = rf_regressor.predict(X_test)

train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)

# Linear Regresssion
lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)
test_mse_lr = mean_squared_error(y_test, y_test_pred_lr)
train_r2_lr = r2_score(y_train, y_train_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)

# Comparison of Model Performanc
print("Decision Tree Regression Performance:")
print("Training set Mean Squared Error (MSE):", train_mse_dt)
print("Test set Mean Squared Error (MSE):", test_mse_dt)
print("Training set R-squared (R2):", train_r2_dt)
print("Test set R-squared (R2):", test_r2_dt)

print("\nRandom Forest Regression Performance:")
print("Training set Mean Squared Error (MSE):", train_mse_rf)
print("Test set Mean Squared Error (MSE):", test_mse_rf)
print("Training set R-squared (R2):", train_r2_rf)
print("Test set R-squared (R2):", test_r2_rf)

print("\nLinear Regression Performance:")
print("Training set Mean Squared Error (MSE):", train_mse_lr)
print("Test set Mean Squared Error (MSE):", test_mse_lr)
print("Training set R-squared (R2):", train_r2_lr)
print("Test set R-squared (R2):", test_r2_lr)

new_data = pd.read_csv('new_data.csv')


# Make predictions with the trained models
new_data_predictions_dt = dt_regressor.predict(new_data)
new_data_predictions_rf = rf_regressor.predict(new_data)
new_data_predictions_lr = lr.predict(new_data)

new_data['Predicted_repay_fail_DT'] = new_data_predictions_dt
new_data['Predicted_repay_fail_RF'] = new_data_predictions_rf
new_data['Predicted_repay_fail_LR'] = new_data_predictions_lr

# Display the new data with predictions
print(new_data.head())


# Cross-validation for Randm Forest Regressor
cv_scores_rf = cross_val_score(rf_regressor, X, y, cv=5, scoring='r2')
print("\nRandom Forest Cross-Validation R2 Scores:", cv_scores_rf)
print("Average Cross-Validation R2 Score:", cv_scores_rf.mean())

# Cross-validation for Decision Tree Regressor
cv_scores_dt = cross_val_score(dt_regressor, X, y, cv=5, scoring='r2')
print("\nDecision Tree Cross-Validation R2 Scores:", cv_scores_dt)
print("Average Cross-Validation R2 Score:", cv_scores_dt.mean())

# Cros-validation for Linear Regression
cv_scores_lr = cross_val_score(lr, X, y, cv=5, scoring='r2')
print("\nLinear Regression Cross-Validation R2 Scores:", cv_scores_lr)
print("Average Cross-Validation R2 Score:", cv_scores_lr.mean())


```
</pre>
