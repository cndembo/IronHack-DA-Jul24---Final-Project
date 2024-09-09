#Libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train_ml(bank_df):
    
    #label_encoding
    bank_df['Credit_Mix'] = bank_df['Credit_Mix'].astype(str)

    label_encoder = LabelEncoder()
    bank_df['Payment_Amount_encoded'] = label_encoder.fit_transform(bank_df['Payment_of_Min_Amount'])
    # Type_of_Loan_fixed_encoded - One-hot enconding
    bank_df['Type_of_Loan_fixed_encoded'] = label_encoder.fit_transform(bank_df['Type_of_Loan_fixed'])

    #manual Label encoding
    credit_mix_mapping = {'Bad': 0, 'Standard': 1, 'Good': 2}
    credit_score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
    payment_behaviour_mapping = {'Low_spent_Small_value_payments':0,
                                'Low_spent_Medium_value_payments':1,
                                'Low_spent_Large_value_payments':2,
                                'High_spent_Small_value_payments':3,
                                'High_spent_Medium_value_payments':4,
                                'High_spent_Large_value_payments':5}

    # Apply the mappings
    bank_df['Credit_Mix_encoded'] = bank_df['Credit_Mix'].map(credit_mix_mapping)
    bank_df['Credit_Score_encoded'] = bank_df['Credit_Score'].map(credit_score_mapping)
    bank_df['Payment_Behaviour_encoded'] = bank_df['Payment_Behaviour'].map(payment_behaviour_mapping)

    #Distinguish target and features data frames
    target = bank_df['Credit_Score_encoded']
    #'Unnamed: 0'
    features = bank_df.drop(columns = ['ID','Age','Customer_ID', 'Name', 'Type_of_Loan', 'Month', 'Months', 'Years', 
                                    'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score', 
                                    'Type_of_Loan_fixed', 'Credit_Score_encoded',
                                    'Payment_Amount_encoded', 'Type_of_Loan_fixed_encoded', 'Credit_Mix_encoded', 'Payment_Behaviour_encoded'])

    #no features engeneering to apply
    #split train/test
    print(f"\nSplit Train/Test Data!")
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 0)

    print(f"\nNormalizing!")
    #apply feature scalling
    normalizer = MinMaxScaler()
    #standerdizer = StandardScaler()
    #normalizer.fit(X_train)

    #Initialize Classification models
    rfc = RandomForestClassifier(random_state=0)
    knn = KNeighborsClassifier()
    #lrc = LogisticRegression()
    dtc = DecisionTreeClassifier()
    ada = AdaBoostClassifier()
    gdb = GradientBoostingClassifier()

    print("\nFiting models (please hold!)")
    #fit models
    rfc.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    #lrc.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    ada.fit(X_train, y_train)
    gdb.fit(X_train, y_train)


    print("\nPredicting!")
    #prediction
    rfc_pred = rfc.predict(X_test)
    knn_pred = knn.predict(X_test)
    #lrc_pred = lrc.predict(X_test)
    dtc_pred = dtc.predict(X_test)
    ada_pred = ada.predict(X_test)
    gdb_pred = gdb.predict(X_test)

    print("\nEvaluating predictions accuracy")
    # Accuracy Evaluation
    print("\nRandom Forest evaluation")
    accuracy = accuracy_score(y_test, rfc_pred)
    print(f'Accuracy: {accuracy}')

    print("\nKNeighbors evaluation")
    accuracy = accuracy_score(y_test, knn_pred)
    print(f'Accuracy: {accuracy}')

    # print("\nLogistic Regression evaluation")
    # accuracy = accuracy_score(y_test, lrc_pred)
    # print(f'Accuracy: {accuracy}')

    print("\nDecisionTree Classifier evaluation")
    accuracy = accuracy_score(y_test, dtc_pred)
    print(f'Accuracy: {accuracy}')

    print("\nAdaBoost Classifier evaluation")
    accuracy = accuracy_score(y_test, ada_pred)
    print(f'Accuracy: {accuracy}')

    print("\nGradientBoost Classifier evaluation")
    accuracy = accuracy_score(y_test, gdb_pred)
    print(f'Accuracy: {accuracy}')

    #Save the selected model
    joblib.dump(rfc, 'trained_score_model_rfc.pkl')

    print("ML model saved!")