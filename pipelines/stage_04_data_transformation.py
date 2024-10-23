import configparser
import os
import sys
import pandas as pd
import os.path as path

# Set parent directory for imports
parent_directory = os.path.abspath(path.join(__file__, "../../"))
sys.path.append(parent_directory)
# Import custom components
from src.components.data_cleaning import DataClean
from src.components.data_transformation import DataTransformation
from src.utils.common import *

# Constants and Configurations
STAGE_NAME = "Data Transformation"
config = configparser.RawConfigParser()



class DataTransformationPipeline:
    def __init__(self):
        self.transform_obj = DataTransformation()

    def main(self):
        try:
            # Read the cleaned dataset
            df = self.transform_obj.read_data()

            # Data Cleaning and Preparation
            df = self.clean_data(df)

            # Perform feature engineering on 'sub_area'
            df = self.engineer_sub_area(df)

            # Select and scale final features
            sel_data = self.scale_features(df)

            # Save transformed data to CSV
            self.transform_obj.save_to_csv(sel_data)

        except Exception as e:
            print(f"Error in {STAGE_NAME}: {e}")
            raise e

    def clean_data(self, df):
        """Cleans the dataset by dropping unnecessary columns and duplicates."""
        print("Cleaning Data...")
        df = df.drop(columns=["index", "company_name", "township"], axis=1)
        df = df.drop_duplicates()

        binary_cols = df.iloc[:, 4:].columns.to_list()  # Get binary columns
        df = self.transform_obj.encode_categorical_variables(df, binary_cols)
        return df

    def engineer_sub_area(self, df):
        """Feature engineering on 'sub_area'."""
        print("Engineering 'sub_area' feature...")

        # Contribution of sub-areas
        df_sa_count = df.groupby("sub_area")["price"].count().reset_index()\
            .rename(columns={"price": "count"})\
            .sort_values("count", ascending=False)\
            .reset_index(drop=True)

        df_sa_count["sa_contribution"] = df_sa_count["count"] / len(df)

        # Feature engineering for 'sub_area' and encoding
        df = self.transform_obj.fea_eng_sa(df_sa_count, "sub_area", df, 7)
        encoded_df = self.transform_obj.onehot_end(df, "sub_area")
        
        # Replace 'sub_area' with its encoded columns
        df = pd.concat([df.drop('sub_area', axis=1), encoded_df], axis=1)
        
        return df

    def scale_features(self, df):
        """Scales the selected features."""
        print("Scaling Features...")

        # Selected features for the final dataset
        features = [
            'surface', 'n_bhk', 'pool', 'club_house', 'mall',
            'park', 'gym', 'sub_area_baner', 'sub_area_bavdhan',
            'sub_area_bt kawade rd', 'sub_area_handewadi',
            'sub_area_hinjewadi ', 'sub_area_kharadi', 'sub_area_mahalunge',
            'sub_area_nibm', 'sub_area_other',
            'sub_area_wadgaon sheri '
        ]

        # Select and scale final data
        sel_data = df[features + ["price"]].copy()
        sel_data = self.transform_obj.data_scale(sel_data, "surface")
        return sel_data

if __name__ == '__main__':
    try:
        print(f">>>>>> Stage started <<<<<< : {STAGE_NAME}")
        pipeline = DataTransformationPipeline()
        pipeline.main()
        print(f">>>>>> Stage completed <<<<<< : {STAGE_NAME}")
    except Exception as e:
        print(e)
        raise e