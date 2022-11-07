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
![image](https://user-images.githubusercontent.com/99802770/200241590-eb731200-af0a-494b-942f-aef092d018ed.png)

Classes balance

![image](https://user-images.githubusercontent.com/99802770/200241698-7016bea4-1d96-40b2-ad4d-fb40b1425622.png)


Checking Target Dependencies on Features

![image](https://user-images.githubusercontent.com/99802770/200241743-1e883ea8-4dc7-422a-857f-19307391ca89.png)

Checking the distribution of numeric values

![image](https://user-images.githubusercontent.com/99802770/200241827-333bcc1a-b4d4-4956-963c-76ac8e02cf85.png)

for more EDA see the [EDA.ipynb](https://github.com/zhuDmi/Learning_Analitics/blob/master/notebooks/EDA.ipynb)

## Choose the models

For compare baselines we are choose 2 models: Catboost and Lightgbm. Base metrics is F1

![Снимок экрана 2022-11-07 в 12 51 52](https://user-images.githubusercontent.com/99802770/200241913-cbb50e7a-05d8-4e49-9862-6a392ead7a27.png)

Hyperparameter fitting done with Optuna

![Снимок экрана 2022-11-07 в 12 56 05](https://user-images.githubusercontent.com/99802770/200241989-7d73af7f-1047-44fe-8f09-3871c955ba17.png)


Stacking done. GaussianNB is chosen as the metamodel

![Снимок экрана 2022-11-07 в 12 58 34](https://user-images.githubusercontent.com/99802770/200242048-b02d33ab-11dd-4461-ab04-a12666d2b030.png)


## Feature importance

![image](https://user-images.githubusercontent.com/99802770/200242140-9427541c-cbac-4a60-9a40-a79d2e86375f.png)


## Model performance

Solution can be reproduced on GPU.

GPU characteristics: Tesla T4.

Time to inference is 0:00:00.758324.
