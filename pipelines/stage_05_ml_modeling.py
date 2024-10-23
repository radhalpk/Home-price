import configparser
config = configparser.RawConfigParser()
import os.path as path
import pandas as pd
import sys
import os
import mlflow
import mlflow.sklearn
from IPython.display import display
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import mlflow

parent_directory = os.path.abspath(path.join(__file__ ,"../../"))
sys.path.append(parent_directory)

from src.components.data_cleaning import DataClean
from src.utils.common import *

from src.components.model_trainer import ModelTrainer

from src.constants import *
from src.utils import *
import configparser
config = configparser.RawConfigParser()


config.read(path.abspath(path.join(__file__ ,"../../src/config/config.ini")))

config_data=config["DATA"]
MODEL_DIR = config.get('DATA', 'model_dir')
PROCESSED_DATA_DIR=config.get('DATA', 'processed_data_dir')


print ("*********",MODEL_DIR)


STAGE_NAME = "MODEL TRAINING"


class MachineLearningModelingPipeline:
    def __init__(self):
        self.config = config.read(CONFIG_FILE_PATH)

    def main(self):
        try:
           
            model_trainer_obj=ModelTrainer()
            data = pd.read_csv(PROCESSED_DATA_DIR) # # read the final csv data
            data = data.sort_values("surface").reset_index(drop=True)

            X = data.iloc[:, :-1] # # Selecting the feature matrix and target vector
            y = data["price"]

        
            rs = 118 # # Random sate for data splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=rs) 
            #print("****",y_test)
            
            #### Find BEST MODEL#################
            score_df=model_trainer_obj.find_best_model(X_train,y_train,X_test,y_test)
            display(score_df)
            
            best_model_row = score_df.loc[score_df['test_r2'].idxmax()]
            best_model_name = best_model_row['model']
            best_model_params = best_model_row['best_parameters']
            best_model_score = best_model_row['test_r2']
            print(f"Best Model: {best_model_name}")
            print(f"Best Parameters: {best_model_params}")
            print(f"Best Score: {best_model_score}")
            # # ## Regresiion models - MODEL BUILDING  ###
            model_reg= model_trainer_obj.final_model(X_train, X_test, y_train, y_test, best_model_name, best_model_params)  
            print("Regression Model Executed")
            
            
           
                       
        except Exception as e:
            raise e


    
if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
        obj = MachineLearningModelingPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e