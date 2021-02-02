## Classification Workflow


### 1. Define the problem
* Audience: HR of the tech company
* Use case: Find qualified candidates who are actually looking for a job change


### 2. EDA
* Data acquisition
* Data cleaning
* Visualization; pair plots, correlation, classes balance 
* Preprocessing: scaling/standardizing


### 3. Train/validation/test set or Cross-validation

### 4. Metrics
* Confusion matrix
* Precision: when there's a cost to being wrong
* **Recall: catch as many as possible**
* F1: balance between precision and recall
* Accuracy: when classes are balanced
* ROC AUC: good for comparing models (taking all into account)
* Log loss: when want to penalize for being really far off


### 5. Baseline
* Dummy classifier: assign everything to the majority class (use this for comparison)
* Try different simple models:
  * kNN
  * Logistic
  * Naive Bayes (bernoulli, gaussian, multinomial)
  * Decision trees
  * Random Forest
  * XGBoost
* Check performance on chosen metrics and compare with ROC AUC


### 6. Select a model to refine
* Overfitting/underfitting? --> tune hyperparameters (Grid search)
* Feature engineering? Take some features out?
* Balance the classes? --> under/oversampling, synthetic
* Check confusion matrix --> adjust threshold
* Scaling/standarrdization
  * kNN
  * Logistic with regularization
  * SVM  
* Other techniques:
  * Ensembling: putting 2 or more models together (help with overgitting but lose interpretability)
* Test model performance (iteratively): use validation data


### 7. Finalize model
* Determined: features, preprocessing, imbalance handling strategy, hyperparameters
* Run on test/hold out data

### 8. Interpret and communicate
* Interesting or unexpected takeaways
* Relate to the use case
* Comparisons between train and test performance
* Meaning of coefficients
* Feature importance (CART model)
* Outliers in the predictions: why did the model get it wrong?

  
