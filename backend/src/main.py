"""

"""

import warnings
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
from backend.src.evaluate.evaluate_model import *
warnings.filterwarnings('ignore')
app = FastAPI()

config_path = '../config/params.yml'

class InsuranceCustomer(BaseModel, config_path):

# TODO continuous
