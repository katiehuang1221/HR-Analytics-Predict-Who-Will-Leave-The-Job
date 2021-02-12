# Project 3: Classification - Smart HR
###### Weeks 4, 5, 6.5

## Backstory

Company X has conducted a series of training courses and is planning to hire Data Scientists from those who have successfully completed them. However, there are more than 20,000 enrollees and a large porition of them may not be considering about switching to new jobs. Therefore, in order to help HR of company X to find talent more efficiently, the model "Smart HR" is developed. Based on the enrollee's information (such as gender, education, credentials and current company), Smart HR will calculate the probability of the enrollee looking for a new job.

## Impact

Immediate benefit for the company is reduce the time and cost for reaching out to/interviewing candidates. In the long run, the model prediction can help with retaining talent by analyzing factors of employee decision.

## Web App Access

(Platform: Streamlit) Python script can be found [here](https://github.com/katiehuang1221/onl_ds5_project_3/blob/main/py/main.py).
* Visualization and statistics of enrollee data structure
* Feature filters for the enrollees
* Model performance
* Customized metrics
* Final list of candidates

![alt text](https://github.com/katiehuang1221/onl_ds5_project_3/blob/main/img/streamlit_1.png)
![alt text](https://github.com/katiehuang1221/onl_ds5_project_3/blob/main/img/streamlit_2.png)
![alt text](https://github.com/katiehuang1221/onl_ds5_project_3/blob/main/img/streamllit_3.png)

#### Demo
![embed](https://github.com/katiehuang1221/onl_ds5_project_3/blob/main/streamlit_demo_fast.gif)


### Data

 * **source**: [Kaggle, HR Analytics](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)
 * **storage**: flat files, SQL


### Target
<ins>*Is the enrollee looking for a new job?*</ins>

* Yes
* No

### Features

<ins>*Current Status*</ins>
  - Gender
  - Current city
  - Current city development index

<ins>*Education*</ins>
  - Major
  - Education level

<ins>*Current Company*</ins>
  - Company type
  - Company size
  
<ins>*Credentials*</ins>
  - Enrolled courses
  - Training hours
  - Relevant experience
  - Experience (years)
  
  
 

  

### Skills

 * SQL
 * `numpy`, `pandas`
 * `statsmodels`, `scikit-learn`
 * `matplotlib`, `seaborn`, `altair`
 * Streamlit


### Analysis
*Models can be found in these Jupyter Notebooks:*
[Logistic Regression, kNN, Naive Bayes](https://github.com/katiehuang1221/onl_ds5_project_3/blob/main/notebook/08_refine_models_1.ipynb),
[Tree based and others](https://github.com/katiehuang1221/onl_ds5_project_3/blob/main/notebook/09_refine_models_2.ipynb).
#### Classification Models

 * Logistic Regression
 * kNN
 * Naive Bayes
 * Decision Trees
 * Random Forests
 * Extra Trees
 * XGBoost
 * Other ensemble methods
    - Voting Classifier (Hard, Soft)
    - Bagging Classifier
    - Boosting (AdaBoost, Gradient Boost)
    - Stacking
 
#### Others

 * Preprocessing & Feature Engineering
    - `StandardScaler`,`PolynomialFeatures`
    - `OneHotEncoder`,`LabelEncoder`
    - `KNNImputer`
    
 * Cross-Validation & hyperparameter Tuning
    - `GridSearchCV`

 


