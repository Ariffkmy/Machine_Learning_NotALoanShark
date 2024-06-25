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

