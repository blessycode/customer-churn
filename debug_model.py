import pickle
import pandas as pd

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load sample data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df.drop('customerID', axis=1)
df = df.drop('Churn', axis=1)

print("Sample data shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nColumn dtypes:")
print(df.dtypes)
print("\nSample row:")
print(df.iloc[0])
print("\nFirst few columns:")
print(df.head(2))

# Try to get feature names from model
if hasattr(model, 'get_feature_names_out'):
    print("\nModel feature names:")
    print(model.get_feature_names_out())
else:
    print("\nModel doesn't have get_feature_names_out method")

# Check if model is a pipeline
if hasattr(model, 'named_steps'):
    print("\nModel is a pipeline with steps:")
    for name, step in model.named_steps.items():
        print(f"  - {name}: {type(step)}")
