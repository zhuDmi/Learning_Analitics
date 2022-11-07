"""
Model for predicting student performance
"""

import warnings

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.evaluate.evaluate_model import *
from src.pipelines.pipeline import pipeline_training
from src.train.metrics import load_metrics
from src.evaluate.evaluate_model import pipeline_evaluate

warnings.filterwarnings('ignore')
app = FastAPI()

config_path = '/config/params.yml'


class StudentParameters(BaseModel):
    """
    Features for obtaining model results
    """
    ST_YEAR: int
    SEMESTER: int
    TYPE_NAME: str
    CHOICE: int
    DISC_DEP: str
    STUD_GENDER: str
    CITIZENSHIP: str
    EXAM_TYPE: str
    EXAM_SUBJECT_1: str
    EXAM_SUBJECT_2: str
    EXAM_SUBJECT_3: str
    ADMITTED_EXAM_1: int
    ADMITTED_EXAM_2: int
    ADMITTED_EXAM_3: int
    ADMITTED_SUBJECT_PRIZE_LEVEL: str
    REGION_ID: str
    KURS: int
    PRIZNAK: str
    Training_period: int


@app.get('/hello')
def welcome() -> dict[str, str]:
    """
    Welcome print function
    :return: None
    """
    return {'message': 'Hello student'}


@app.post('/train')
def training() -> dict[str, dict]:
    """
    Model training, metrics logging
    :return: metrics
    """
    pipeline_training(config_path=config_path)
    metrics = load_metrics(config_path=config_path)
    return {'metrics': metrics}


@app.post('/predict_input')
def prediction_input(student: StudentParameters) -> str:
    """
    Model prediction from input data
    :param student: student parameters
    :return: result of model prediction
    """
    features = [student.ST_YEAR,
                student.SEMESTER,
                student.TYPE_NAME,
                student.CHOICE,
                student.DISC_DEP,
                student.STUD_GENDER,
                student.CITIZENSHIP,
                student.EXAM_TYPE,
                student.EXAM_SUBJECT_1,
                student.EXAM_SUBJECT_2,
                student.EXAM_SUBJECT_3,
                student.ADMITTED_EXAM_1,
                student.ADMITTED_EXAM_2,
                student.ADMITTED_EXAM_3,
                student.ADMITTED_SUBJECT_PRIZE_LEVEL,
                student.REGION_ID,
                student.KURS,
                student.PRIZNAK,
                student.Training_period]
    cols = ['ST_YEAR',
            'SEMESTER',
            'TYPE_NAME',
            'DEBT',
            'СHOICE',
            'DISC_DEP',
            'STUD_GENDER',
            'CITIZENSHIP',
            'EXAM_TYPE',
            'EXAM_SUBJECT_1',
            'EXAM_SUBJECT_2',
            'EXAM_SUBJECT_3',
            'ADMITTED_EXAM_1',
            'ADMITTED_EXAM_2',
            'ADMITTED_EXAM_3',
            'ADMITTED_SUBJECT_PRIZE_LEVEL',
            'REGION_ID',
            'KURS',
            'PRIZNAK',
            'Training_period']
    data = pd.DataFrame(features, columns=cols)

    predictions = pipeline_evaluate(config_path=config_path, dataset=data)[0]
    result = (
        {"You are more likely to be in debt"}
        if predictions == 1
        else {"You are less likely to be in debt"}
        if predictions == 0
        else "Error result")

    return result


if __name__ == '__main__':
    # run server host
    uvicorn.run(app, host="127.0.0.1", port=80)
