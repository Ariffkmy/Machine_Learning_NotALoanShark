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
