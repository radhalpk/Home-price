import configparser
config = configparser.RawConfigParser()
import os.path as path
import pandas as pd
import sys
import os

parent_directory = os.path.abspath(path.join(__file__ ,"../../"))
sys.path.append(parent_directory)

from src.components.data_cleaning import DataClean
from src.utils.common import *


STAGE_NAME = "Data Cleaning"
cleaning_obj=DataClean()

class DataCleaningPipeline:
    def _init_(self):
        pass

    def main(self):
        try:
            # Read the initial dataset
            dfr = cleaning_obj.read_data()  
            
            ### DATA CLEANING ###
            
            # Rename columns and drop values
            dfr = cleaning_obj.rename_col(dfr, "Propert Type", "Property Type")
            df_norm = cleaning_obj.drop_val(dfr, "Property Type", "shop")
            
            # Clean and process specific columns
            df_norm.loc[:, "Property Type"] = df_norm["Property Type"].apply(cleaning_obj.splitSums)
            df_norm.loc[:, "Property Area in Sq. Ft."] = df_norm["Property Area in Sq. Ft."].apply(lambda x: cleaning_obj.splitSums(x, False))
            
            # Outlier detection and removal
            x_prt = df_norm['Property Type']
            prt_up_lim = cleaning_obj.computeUpperFence(x_prt)
            df_norm = df_norm.loc[x_prt <= prt_up_lim]  # Remove outliers based on Property Type
            df_norm.drop(index=df_norm.loc[df_norm["Property Type"] == 7].index, inplace=True)  # Drop 7 BHK outliers
            df_norm.drop(index=86, inplace=True)  # Drop row 86, an identified outlier
            
            # Price cleaning and dropping unnecessary columns
            df_norm.loc[:, "Price in lakhs"] = pd.to_numeric(df_norm["Price in lakhs"], errors='coerce')
            df_norm.drop(columns=["Price in lakhs"], axis=1)
            
            # Handling missing values
            cleaning_obj.compute_fill_rate(df_norm)
            df_norm[["Sub-Area", "TownShip Name/ Society Name", "Total TownShip Area in Acres" ]]\
                .sort_values("Sub-Area").reset_index(drop=True)        # Total TownShip Area in Acres
            df_norm = cleaning_obj.drop_empty_axis(df_norm, minFillRate=0.5)  # Drop columns with less than 50% filled
            
            ### Regularizing categorical columns ###
            binary_cols = df_norm.iloc[:, -7:].columns.to_list()
            df_norm = df_norm[df_norm["Price in Millions"] < 80]  # Keep only rows with target price less than 80
            binary_cols = cleaning_obj.reg_catvar(df_norm, binary_cols)  # Regularize binary categorical variables
            
            # Handle multicategorical columns
            obj_cols = df_norm.select_dtypes(include="object").columns.to_list()
            multiCat_cols = list(set(obj_cols) ^ set(binary_cols))
            multiCat_cols = cleaning_obj.reg_catvar(df_norm, multiCat_cols)
            
            # Drop unnecessary columns
            df_norm.drop(columns=["Location", "Price in Millions"], axis=1, inplace=True)
            
            # Rename columns
            df_norm.columns = ["index", "sub_area", "n_bhk", "surface", "price", 
                               "company_name", "township", "club_house", "school", 
                               "hospital", "mall", "park", "pool", "gym"]
            
            # Save cleaned data
            cleaning_obj.save_to_csv(df_norm)

        except Exception as e:
            print(f"Error in Data Cleaning Pipeline: {e}")
            raise e




    
if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
        obj = DataCleaningPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e