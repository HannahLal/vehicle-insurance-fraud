# Capstone Project: Vehicle Insurance Claims Fraud Detection System
========================================================================================
## Introduction
========================================================================================

Insurance fraud is a prevalent and costly challenge for both policyholders and insurance companies. 
Fraudulent activities may encompass "false claims," "misrepresentation of information," or "organized fraud schemes." 
Detecting and preventing these activities is crucial for minimizing financial losses for issuers as well as 
safeguarding policyholders. Given its significant impact, insurance fraud detection has emerged as a prominent 
research area in data science and machine learning. This capstone project aims to concentrate on identifying and 
preventing fraudulent or misleading insurance claims using the dataset available at Kaggle. The objective is to develop 
algorithms and models that can automatically detect suspicious activities through historical data analysis. 

========================================================================================
## Project Organization
========================================================================================
    
    ├── README.md                              <- The main README document for developers utilizing this project.
    |
	├── presentations				        <- The folder containing all notebooks for every sprint of the project
    |   ├── EDA_Presentation.pdf               <- Sprint 1 presentation of the project on Performing EDA on raw dataset
	|	├── ...						           <- Final presentation of the project. To be completed ... 
    |
    ├── notebooks							   <- The folder containing all notebooks for every sprint of the project
    |   ├── data                               <- The folder containing all dataset (clean and raw) used throughout the whole project
	|   |   ├── fraud_oracle.csv			   <- The raw dataset of vehicle fraud insurance (found at https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data)
    |   ├── exploratory_data_analysis.ipynb    <- Project notebook 1 - data preparation and exploration
    |   ├── ...						           <- To be implemented ...
    |
    ├── models                                 <- The folder containing all trained models to be saved. To be completed ...
 

========================================================================================
## Table of contents
========================================================================================

- Vehicle Insurance Dataset 
- Dataset Access Link, Preparation, Necessary Cleaning and Exploratory Data Analysis
- Basline Fraud Detection System (To be completed ...)
- Modeling (To be completed ...)
- Conclusions (To be completed ...)

========================================================================================
## Vehicle Insurance Dataset
========================================================================================

The dataset (fraud_oracle.csv) includes the following fields:

	- Month: Month of year the accident happened (text)
	- WeekOfMonth : Week number in a month the accident happened (number)
	- DayOfWeek : Day of the week the accident happened (text)
	- Make : Brand of the vehicle (text)
	- AccidentArea : where the accident happened (text) (Urban/Rural)
	- DayOfWeekClaimed : Day of the week in which the claim was made (text)
	- MonthClaimed : The month in which the claim was made (text)
	- WeekOfMonthClaimed : The week number in the month during which the claim was made (number)
	- Sex : Gender of the vehicle owner (text) (Male/Female)
	- MaritalStatus : Marital status of the vehicle owner (text) (Married/Single/Widow/Divorced)
	- Age : Age of the vehicle owner (number)
	+ Fault : Fault of the accident(text)
		- Policy Holder : The fault of the accident is on Policy holder
		- Third Party : The fault of the accident is on the third party
	+ PolicyType : Indicates the type of coverage for the policy, specifying whether it covers liability, collision, or all perils for each vehicle type (text)
		- 'Sport - Liability' : Covers only liability for Sport vehicle category
		- 'Sport - Collision' : Covers only collision for Sport vehicle category
		- 'Sedan - Liability' : Covers only liability for Sedan vehicle category
		- 'Utility - All Perils' : Covers all perils for Utility vehicle category 
		- 'Sedan - All Perils' : Covers all perils for Sedan vehicle category
		- 'Sedan - Collision' : Covers only collision for Sedan vehicle category
		- 'Utility - Collision' : Covers only collision for Utility vehicle category
		- 'Utility - Liability' : Covers only liability for Utility vehicle category
		- 'Sport - All Perils' : Covers all perils for Sport vehicle category
	+ VehicleCategory : Describes the category of the insured vehicle (text)
		- Sport : it is a sports car
		- Utility : it is a utility vehicle
		- Sedan : it is a sedan car
	+ VehiclePrice : The price range of the insured car (text)
		- 'less than 20000' 
		- '20000 to 29000' 
		- '30000 to 39000'  
		- '40000 to 59000'  
		- '60000 to 69000' 
		- 'more than 69000'
	- FraudFound_P : Probability of a fraud being found (number) (can be either 0 or 1)
	- PolicyNumber : A unique identifier for each insurance policy (number)
	- RepNumber : Represents the identifier or code for the insurance representative or agent associated with the policy (number)
	- Deductible : Refers to the amount that the policyholder has to pay out of pocket before the insurance coverage takes effect (number)
    - DriverRating : Indicates the rating or risk associated with the driver insured under the policy (number) (1 to 4)
	+ Days_Policy_Accident : The number of days since the last accident covered by the policy (text)
		- 'none' : Zero days since the last accident could mean the first time this car has been involved in an accident
		- '1 to 7' : Last accident with the same car was in less than a week ago
		- '8 to 15' : Last accident was any where between last two weeks and last week
		- '15 to 30' : Last accident happened more than two weeks ago but less than a month ago
		- 'more than 30' : Last accident happened more than a month ago
	+ Days_Policy_Claim : The number of days since the last insurance claim was filed under the policy (text)
		- 'none' : Zero days since the last insurance claim was filed
		- '8 to 15' : The last insurance claim was filed any time between at least a week ago and at most two weeks ago
		- '15 to 30' : The last insurance claim was filed any time between at least 2 weeks ago and at most a month ago 
		- 'more than 30' : The last insurance claim was filed any time at least a month ago
	+ PastNumberOfClaims : Represents the total number of claims filed by the policyholder in the past (text)
		- 'none' : No claims filed by the policyholder in the past
		- '1' : The policyholder has a history of 1 claim being filed in the past 
		- '2 to 4' : The policyholder has a history of filing 2 to 4 claims in the past
		- 'more than 4' : The policyholder has a history of filing at least 5 claims (and more) in the past
    + AgeOfVehicle : Age of the vehicle involved in the accident (text)	
		- 'new' : The vehicle was new (less than a year)
		- '2 years' : The vehicle was at most 2 years old
		- '3 years' : The vehicle was at least 2 years old and at most 3 years old
		- '4 years' : The vehicle was at least 3 years old and at most 4 years old
		- '5 years' : The vehicle was at least 4 years old and at most 5 years old
		- '6 years' : The vehicle was at least 5 years old and at most 6 years old
        - '7 years' : The vehicle was at least 6 years old and at most 7 years old
        - 'more than 7' : The vehicle was at least 7 years old maybe even older
	+ AgeOfPolicyHolder : The age range of the policyholder (text)
		- '16 to 17' : The policyholder is between 16 and 17 years old
		- '18 to 20' : The policyholder is between 18 and 20 years old
		- '21 to 25' : The policyholder is between 21 and 25 years old
		- '26 to 30' : The policyholder is between 26 and 30 years old
		- '31 to 35' : The policyholder is between 31 and 35 years old
		- '36 to 40' : The policyholder is between 36 and 40 years old
		- '41 to 50' : The policyholder is between 41 and 50 years old
		- '51 to 65' : The policyholder is between 51 and 65 years old
		- 'over 65' : The policyholder is at least over 65 years old
	- PoliceReportFiled : Indicates whether a police report was filed in case of an accident or claim (text) (Yes/No)
	- WitnessPresent : Indicates whether any witnesses were present when the accident happened (text) (Yes/No)
	- AgentType : Describes the type or category of the insurance agent associated with the policy (text) (External/Internal)
	+ NumberOfSuppliments : Indicates the number of additional supplements or endorsements to the policy (text)
		- 'none' : There were no additional supplements or endorsements to the policy
		- '1 to 2' : There were 1 or 2 additional supplements or endorsements to the policy
		- '3 to 5' : There were 3 to 5 additional supplements or endorsements to the policy
		- 'more than 5' : There were at least more than 5 additional supplements or endorsements to the policy
	+ AddressChange_Claim : Indicates whether there has been a change in the policyholder's address following a claim (text)
		- 'no change' : There has been no change in the policyholder's address following the claim
		- 'under 6 months' : The policyholder's address has been changed for under 6 months following the claim
		- '1 year' : The policyholder's address has been changed for at least 6 months and at most a year following the claim
		- '2 to 3 years' : The policyholder's address has been changed for at least 2 years and at most 3 years following the claim
		- '4 to 8 years'  : The policyholder's address has been changed for at least 4 years and at most 8 years following the claim
	+ NumberOfCars : Indicates the number of cars involved in the accident (text)
		- '1 vehicle' : Only 1 vehicle was involved in the car accident
		- '2 vehicles' : 2 vehicles were involved in the accident
		- '3 to 4' : 3 to 4 cars were involved in the accident
		- '5 to 8' : 5 to 8 cars were involved in the accident
		- 'more than 8' : More than 8 cars were involved in the accident
	- Year : The year the accident happened (number) (1994/1995/1996)
	+ BasePolicy : Describes the foundational or base insurance policy that may include standard coverage (text)
		- 'Liability' : The base Insurance policy includes liability
		- 'Collision' : The base Insurance policy includes collision
		- 'All Perils' : The base Insurance policy includes all perils

========================================================================================
## Dataset Access Link, Preparation, Necessary Cleaning and Exploratory Data Analysis
========================================================================================

Link to dataset: https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data 

EDA Insights: 

FraudFound_P:

1. Month, WeekOfMonth, MonthClaimed, WeekOfMonthClaimed, Sex, Dedictible, DriverRating, AddressChange_Claim and Vehicle_Price are positively correlated with our target variable FraudFound_P.

2. DayOfWeek, AccidentArea, DayOfWeekClaimed, MaritalStatud, Age, RepNumber, PoliceReportFiled, WitnessPresent, Year, Age of Vehicle and Number of suppliments are negatively correlated with our target variable FraudFound_P.

Fault:

1. Month, AccidentArea, MonthClaimed, WeekOfMonthClaimed, Sex, PoliceReportFiled, WitnessPresent, AddressChange_Claim, Year, Age of Vehicle and Number of suppliments are positively correlated with Fault variable.

2. WeekOfMonth, DayOfWeek, DayOfWeekClaimed, MaritalStatud, Age, RepNumber, Dedictible, DriverRating, Vehicle_Price and Past Number of Claims are negatively correlated with Fault Variable.

3. Month, MonthClaimed, Year and PastClaims are the predictors with higher correlation with the FraudFound_P target variable. WitnessPresent, Vehicle_Price and PastClaims are the predictors with higher correlation with the Fault variable.

========================================================================================
## Baseline Fraud Detection System
========================================================================================

For our baseline modeling we will be using the logistic regression and decision tree models having the FraudFound_P column as our target variable to predict its values given all the other columns as its independent variables.

Baseline modeling Insights: 

1. Class 0: 
a) Percision: 0.99: Meaning When the model predicts class 0, it is correct 99% of the time
b) Recall: 0.59: Meaning the model captures 59% of the actual instances of class 0
c) F1-Score: 0.74

2. Class 1:
a) Percision: 0.12: Meaning when the model predicts class 1, it is correct only 12% of the time
b) Recall: 0.92: Meaning the model is effective at capturing 92% of the actual instances of class 1
c) F1-Score: 0.22

3. Overall:
a) Accuracy: 0.61: The overall accuracy of the model is 61%
b) Macro avg (precision, recall, F1-score): 0.56, 0.75, 0.48
c) Weighted avg (precision, recall, F1-score): 0.94, 0.61, 0.71
d) Precision and Recall:

4. Percision for class 1: When the model predicts fraud (class 1), it is correct only 12.5% of the time
a) Recall for class 1: The model captures 91.9% of the actual instances of fraud
b) ROC AUC Score:
c) ROC AUC (Receiver Operating Characteristic Area Under the Curve) is a measure of the model's ability to distinguish between positive and negative classes. A score of 0.806 indicates relatively good performance. The closer the score is to 1, the better the model is at distinguishing between the two classes. What this means is that, the model shows a good ability to discriminate between the two classes as indicated by the ROC AUC score.

5. The precision for non-fraud (class 0) remains high, suggesting that when the model predicts non-fraud, it is usually correct.
a) The recall for fraud (class 1) is high, indicating that the model is effective at capturing innstances of fraud.
b) The F1-score provides a balance between precision and recall.


One recommendation to make the model better, is to adjust the model threshold and continue monitoring and fine-tuning the model based on our specific requirements for this application. We can also consider exploring other models or techniques to further improve performance, especifically due to the fact that not being able to detect fraudulent claims is a very costly miss on our side.


========================================================================================
## Modeling
========================================================================================

For model enhancement purposes, since the dataset is highly imbalanced, we will first apply SMOTE method to resample the data, then we will fit this resampled data into our baseline models and finally we will be using hyperparameter optimization and ensemble methods like xgboost, gradient boosting and bagging/random forests to further improve the performance of our models. For our evaluation metrics we will be considering different metrics such as precision, recall, and f1 score and we will be plotting confusion matrix and ROC curves and displaying the classification reports.

Some insights: 

We can conclude that training a logistic regression model and applying LogReg with k-fold cross validation and GridSearchCV on the features set indicates that fault and vehicle_category_sport are the two most indicating features in our logistic regression model. And PolicyType_2 and PolicyType_1 along with Address_Change_Claim and Base_Policy_Liability stand in the second place as contributors to this model. 

========================================================================================
## Conclusion
========================================================================================


Finally, we will compare different trained models with each other and specifically analyse important features for each model and draw some conclusions.

Based on the trained models, Logistic Regression combined with GridSearchCV and StratifiedKFold performed best on the features extracted from the original dataset. 
