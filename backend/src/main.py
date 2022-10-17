"""

"""

import warnings
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
from backend.src.evaluate.evaluate_model import *
from backend.src.pipelines.pipeline import pipeline_training
warnings.filterwarnings('ignore')
app = FastAPI()

config_path = '/Users/dmitry/PycharmProjects/Learning_Analitics/config/params.yml'

pipeline_training(config_path)


