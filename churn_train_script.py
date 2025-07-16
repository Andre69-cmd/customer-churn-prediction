from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import json

# Load and prepare your data
data = pd.read_csv("C:/Users/user/Downloads/archive (1)/telco_churn.csv")

# Drop customerID
data.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Drop missing values
data.dropna(inplace=True)

# Convert 'Yes'/'No' to 1/0
data = data.replace({'Yes': 1, 'No': 0})

# Handle 'No internet service' as 'No' for selected columns
cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in cols:
    data[col] = data[col].replace({'No internet service': 'No'})

# One-hot encoding for categorical variables
data = pd.get_dummies(data)

# Split into features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save feature names for later use in the app
with open("features.json", "w") as f:
    json.dump(list(X_train.columns), f)

# Train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save the model
model.save_model("model.json")
