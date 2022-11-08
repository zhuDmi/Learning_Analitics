# Learning Analytics
A system that allows predicting academic degrees for students with preschool education.
Team: Voronkina Daria, Zhukov Dmitriy, Kuchuganova Svetlana
## Quick start examples
Clone repo and install [requirements.txt](https://github.com/zhuDmi/Learning_Analitics)
```bash
python3 -m venv env #setup enviroment
source env/bin/activate

git clone https://github.com/zhuDmi/Learning_Analitics  # clone
cd Learning_Analitics
pip install -r requirements.txt  # install
```
## EDA
![img_1.png](demo/img_1.png)

Classes balance

![img_2.png](demo/img_2.png)

Checking Target Dependencies on Features

![img_3.png](demo/img_3.png)

Checking the distribution of numeric values

![img_4.png](demo/img_4.png)

for more EDA see the [EDA.ipynb](https://github.com/zhuDmi/Learning_Analitics/blob/master/notebooks/EDA.ipynb)

## Choose the models

For compare baselines we are choose 2 models: Catboost and Lightgbm. Base metrics is F1

![telegram-cloud-photo-size-2-5427116226095725821-x](https://user-images.githubusercontent.com/55249362/200521194-17f2acbf-a27c-4909-9c40-dcd7b77f1e16.jpg)

Hyperparameter fitting done with Optuna

![img_5.png](../../Desktop/temp/Снимок экрана 2022-11-07 в 12.56.05.png)

Stacking done. GaussianNB is chosen as the metamodel

![img_6.png](../../Desktop/temp/Снимок экрана 2022-11-07 в 12.58.34.png)

## Feature importance

![img_5.png](demo/img_5.png)

## Model performance

Solution can be reproduced on GPU.

GPU characteristics: Tesla T4.

Time to inference is 0:00:00.758324.
