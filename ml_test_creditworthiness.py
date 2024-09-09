import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_model():
    # Load the trained model, Preprocess Data and Predict
    model_rfc = joblib.load('trained_score_model_rfc.pkl')
    bank_df = pd.read_csv('test.csv')
    return bank_df, model_rfc

def data_preparation(bank_df):
    features = bank_df.drop(columns = ['ID','Customer_ID','Month','Name',
                                    'Age','SSN','Occupation',
                                    'Type_of_Loan', 'Credit_Mix', 'Credit_History_Age',
                                    'Payment_of_Min_Amount', 'Total_EMI_per_month',
                                    'Payment_Behaviour','Changed_Credit_Limit',
                                    'Num_Credit_Inquiries','Outstanding_Debt','Amount_invested_monthly'])

    features['Annual_Income'] = features['Annual_Income'].str.replace('_','')
    features['Num_of_Loan'] = features['Num_of_Loan'].str.replace('_','')
    features['Num_of_Delayed_Payment'] = features['Num_of_Delayed_Payment'].str.replace('_','')
    features['Num_of_Delayed_Payment'] = features['Num_of_Delayed_Payment'].fillna(0)
    features['Monthly_Balance'] = features['Monthly_Balance'].replace({'__-333333333333333333333333333__': 0})

    features['Annual_Income'] = features['Annual_Income'].astype(float)
    features['Num_of_Loan'] = features['Num_of_Loan'].astype(int)
    features['Num_of_Delayed_Payment'] = features['Num_of_Delayed_Payment'].astype(int)
    features['Monthly_Balance'] = features['Monthly_Balance'].astype(float)
    
    return features

def predict(features, model_rfc):
    score_pred = features
    score_pred.fillna(0, inplace=True)

    #make predictions
    predictions_rfc = model_rfc.predict(score_pred)

    #save predictions
    features['Predictions_RFC'] = predictions_rfc
    pred_mapping = {0:'Unsafe', 1:'Safe', 2:'Safe'}
    features['Prediction_RFC_label'] = features['Predictions_RFC'].map(pred_mapping)

    features.to_csv('Credit_Score_predictions.csv', index=False)
    print("Predictions executed and saved!")

    print("\nPrediction count:")
    print(features['Prediction_RFC_label'].value_counts())

    print("\nML completed. Please check the files on your folder!")
