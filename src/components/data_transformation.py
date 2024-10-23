from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import os
import configparser
config = configparser.RawConfigParser()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
config.read(r'C:\Users\pavan\OneDrive\Desktop\HomePrice_predection (1)\HomePrice\src\config\config.ini')
from src.constants import *
from src.utils import *


import pandas as pd
import pickle


class DataTransformation:
    def _init_(self):
        #self.config = config.read(CONFIG_FILE_PATH)
        self.label_encoders = {} # Dictionary to store the label encoders
    
    # Function to read clean data and return dataframe
    
    def read_data(self):
        return pd.read_csv(config.get('DATA', 'clean_data_dir'))
    
    # Function for performing label encoding for the categorical variables

    def encode_categorical_variables(self,df, cat_vars):
    # Transforming the yes/no to 1/0
        laben = LabelEncoder()
        self.label_encoders = {}
        for col in cat_vars:
            df[col] = laben.fit_transform(df[col])
            self.label_encoders[col] = laben  # Save the encoder for later use
        LABEL_DIR = config.get('DATA', 'label_encoders_dir')
        
        # Save the encoders to a pickle file
        pickle.dump(self.label_encoders, open(LABEL_DIR + 'label_encoder.pkl', 'wb'))
        return (df)
    

    # function for surface area column
    def fea_eng_sa(self,df_count, df_col, df, n):
            sa_sel_col = df_count.loc[df_count["count"]>n, df_col].to_list()
            df[df_col] = df[df_col].where(df[df_col].isin(sa_sel_col), "other")
            return df


    # function to perform one hot encoding
    def onehot_end(self,df,col_name):
        # Dummy variable conversion
        hoten = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_dummy = hoten.fit_transform(df[[col_name]] )
        feature_names=hoten.get_feature_names_out([col_name])
        df_encoded = pd.DataFrame(X_dummy, columns=feature_names)
        # Save the one-hot encoder
        # Save the one-hot encoder
        ONEHOT_DIR = config.get('DATA', 'one_hot_encoder_dir')
        
        # Save the encoders to a pickle file
        pickle.dump(hoten, open(ONEHOT_DIR + 'onehot_encoder.pkl', 'wb'))
        return df_encoded
    
    # fucntion to perform feature selection
    def feat_sel(self,data, corr_cols_list,target,col_name):
    # Price correlation with all other columns
        # Remove the target column from the list (if present)
         # Remove target from the correlation list (if it's present)
        if target in corr_cols_list:
            corr_cols_list.remove(target)

        # Ensure col_name (one-hot encoded columns) doesn't duplicate existing columns
        corr_cols_list = list(set(corr_cols_list).union(set(col_name)))

        # Now, calculate correlations
        corr_list = []  # To store correlations with the target column
        for col in corr_cols_list:
            corr_list.append(round(data[target].corr(data[col]), 2))
        return corr_list


    # function for surface area count
    def feature_sa(self,df, df_col, target,features):
        # Keeping the sub areas' name, their mean price and frequency (count)
        sa_feature_list = [sa for sa in features if "sa" in sa]
        lst = []
        for col in sa_feature_list:
            sa_triger = df[col]==1
            sa = df.loc[sa_triger, df_col].to_list()[0]
            x = df.loc[sa_triger, target]
            lst.append( (sa, np.mean(x), df[col].sum()) )
        return lst
    
    # function to scale the data
    def data_scale(self,data,df_col):
        # Standard scaling for surface
        sc = StandardScaler(with_std=True, with_mean=True)
        data[df_col] = sc.fit_transform(data[[df_col]])
        SCALAR_DIR = config.get('DATA', 'scalar_dir')
        # Save the encoders to a pickle file
        try:
            pickle.dump(sc, open(SCALAR_DIR + 'scalar.pkl', 'wb'))
        except Exception as e:
            print(e)
        return data
    
    def save_to_csv(self,df):
        
        df.to_csv(config.get('DATA', 'processed_data_dir'), index=False)