
from credit_score_py import load_data, wrangling_eda
from ml_train_creditworthiness import train_ml
from ml_test_creditworthiness import load_model, data_preparation, predict

import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    #load and process the data

    file_csv = pd.read_csv("cscore_train.csv", low_memory = False)
    bank_df = pd.DataFrame(file_csv)  
            
    load_data(bank_df)
    bank_df = wrangling_eda(bank_df)

    #Train the ML
    train_ml(bank_df)

    #Test the ML
    bank_df, model_rfc = load_model()
    features = data_preparation(bank_df)
    predict(features, model_rfc)
    
main()
