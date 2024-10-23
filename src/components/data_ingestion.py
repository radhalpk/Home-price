import os
import urllib.request as request
import zipfile
from pathlib import Path
import configparser

from constants import CONFIG_FILE_PATH
config = configparser.RawConfigParser()
import os.path as path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
config.read(r'C:\Users\pavan\OneDrive\Desktop\HomePrice_predection (1)\HomePrice\src\config\config.ini')
from src.constants import *
from src.utils import *

STAGE_NAME = "Data Ingestion"

class DataIngestion:
    def _init_(self):
        # self.config = config.read(CONFIG_FILE_PATH)
        pass
        
        
            
    def download_file(self):
        if not os.path.exists(config.get('DATA', 'local_data_file')):
            filename, headers = request.urlretrieve(
                url = config.get('DATA', 'source_url'),
                filename = config.get('DATA', 'local_data_file')
                
           )
            print(f"{filename} download! with following info: \n{headers}")
        else:
            file_size=os.path.getsize(config.get('DATA','local_data_file'))
            print(f"File already exists of size:{file_size}")