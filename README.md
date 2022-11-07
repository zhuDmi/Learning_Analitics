# Learning Analytics
A system that allows predicting academic degrees for students with preschool education.
Team: Voronkina Daria, Zhukov Dmitriy, Kuchuganova Svetlana
## Quick start examples
Clone repo and install [requirements.txt](https://github.com/zhuDmi/Learning_Analitics)
```bash
git clone https://github.com/zhuDmi/Learning_Analitics  # clone
cd Learning_Analitics
pip install -r requirements.txt  # install
```
## EDA
![img.png](img.png)

Classes balance

![img_1.png](img_1.png)

Checking Target Dependencies on Features

![img_2.png](img_2.png)

Checking the distribution of numeric values

![img_3.png](img_3.png)

for more EDA see the [EDA.ipynb](https://github.com/zhuDmi/Learning_Analitics/blob/master/notebooks/EDA.ipynb)

## Choose the models

For compare baselines we are choose 2 models: Catboost and Lightgbm. Base metrics is F1


![img_4.png](../../Desktop/temp/Снимок экрана 2022-11-07 в 12.51.52.png)

Hyperparameter fitting done with Optuna

![img_5.png](../../Desktop/temp/Снимок экрана 2022-11-07 в 12.56.05.png)

Stacking done. GaussianNB is chosen as the metamodel

![img_6.png](../../Desktop/temp/Снимок экрана 2022-11-07 в 12.58.34.png)

## Feature importance

![img_4.png](img_4.png)

## Model performance

Solution can be reproduced on GPU.

GPU characteristics: Tesla T4.

Time to inference is 0:00:00.758324.
