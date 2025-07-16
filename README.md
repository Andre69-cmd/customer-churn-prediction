# customer-churn-prediction
Streamlit app to predict customer churn using XGBoost

# 🧠 Customer Churn Prediction App
This is a machine learning web application that predicts customer churn (whether a customer is likely to leave) based on customer account information. The model was trained on the popular Telco Customer Churn dataset and deployed using Streamlit.

## 📊 Demo
> You can run this locally using Streamlit (instructions below).
 

## 🔍 Features
- Predict churn risk using contract type, tenure, and monthly charges.
- Trained using an **XGBoost Classifier**.
- Clean data processing and one-hot encoding.
- Custom-trained model with saved features for consistent prediction input.
- Streamlit-powered interactive UI.



## 📁 Project Structure
customer-churn-prediction/
│
├── app.py # Streamlit app
├── model.json # Trained XGBoost model
├── features.json # List of feature names used in training
├── train_model.py # Script to prepare data and train the model
├── README.md # This file
└── data/
└── WA_Fn-UseC_-Telco-Customer-Churn.csv



## ⚙️ How to Run the App

### 🔧 Setup Instructions
1. **Clone this repository** 
        
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction

Install required libraries
(You may want to use a virtual environment.)
pip install -r requirements.txt

If you don't have a requirements.txt, install manually:
pip install streamlit pandas numpy xgboost scikit-learn

Run the training script (if model is not trained yet)
python train_model.py

Launch the Streamlit app
streamlit run app.py


🧠 Model Details
Model: XGBoost Classifier

Input Features: Contract type, tenure, monthly charges (and other encoded features if expanded)

Target: Churn (Yes/No)

Dataset: Telco Customer Churn Dataset


📚 Learning Outcomes
Building end-to-end ML apps using Streamlit

Cleaning real-world data for machine learning

Encoding categorical variables with pandas.get_dummies()

Saving and loading models and features for deployment

✅ To-Do / Improvements
Add more features to UI (gender, senior citizen, internet service type, etc.)

Add model interpretability with SHAP or LIME

Deploy online using Streamlit Cloud or HuggingFace Spaces

🙋‍♂️ Author
Made with ❤️ by Onen Andrew

📄 License
This project is licensed under the MIT License
