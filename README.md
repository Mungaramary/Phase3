# Phase3
## SYRIA TEL CUSTOMER CHURN ANALYSIS
![header](https://github.com/Mungaramary/Phase3/assets/99483846/be65bd9d-7bfd-40cf-a958-a4861e1735e3)

### OVERVIEW
SyriaTel is a telecommunications company in Syria. They have been informed that some of their customers have started to churn, discontinue their service.

This analysis will determine what features will indicate if a customer will ("soon") discontinue their service.

### BUSINESS UNDERSTANDING

Predicting and preventing customer churn represents a huge additional potential revenue source for every business .Customer churn (also known as customer attrition) refers to when a customer ( subscriber, user, etc.) ceases his or her relationsh ip with a company. Online businesses typically treat a customer as churned once a particular amount of time has elapsed since the customer’s last interaction with the site or service. The full cost of churn includes both lost revenue and the marketing costs involved with replacing those customers with new ones. Reducing churn is a key business goal of every online business that also includes the telecom

The ability to predict that a particular customer is at a high risk of churning, while there is still time to do something about it, represents a huge additional potential revenue source for every online business. Besides the direct loss of revenue that results from a customer abandoning the business, the costs of initially acquiring that customer may not have already been covered by the customer’s spending to date. (In other words, acquiring that customer may have actually been a losing investment.) Furthermore, it is always more difficult and expensive to acquire a new customer than it is to retain a current paying customer

### DATA UNDERSTANDING

The dataset is seen to have the following columns with their brief description

state: the state the user lives in 

account length: the number of days the user has this account 

area code: the code of the area the user lives in 

phone number: the phone number of the user

international plan: true if the user has the international plan, otherwise false

voice mail plan: true if the user has the voice mail plan, otherwise false 

number vmail messages: the number of voice mail messages the user has sent

total day minutes: total number of minutes the user has been in calls during the day

total day calls: total number of calls the user has done during the day 

total day charge: total amount of money the user was charged by the Telecom company for calls during the day 

total eve charge: total amount of money the user was charged by the Telecom company for calls during the evening 

total night minutes: total number of minutes the user has been in calls during the night 

total night calls: total number of calls the user has done during the night

total night charge: total amount of money the user was charged by the Telecom company for calls during the night

total intl minutes: total number of minutes the user has been in international calls 

total intl calls: total number of international calls the user has done

total intl charge: total amount of money the user was charged by the Telecom company for international calls

customer service calls: number of customer service calls the user has done

churn: true if the user terminated the contract, otherwise false(left the company)
### EDA 
THE CHURN DISTRIBUTION 

![churn](https://github.com/Mungaramary/Phase3/assets/99483846/1ef9bbb2-6fcb-4ae8-a153-3916b50c1eee)

Distribution of charn based on customer service calls 
![customer_service_call_churn](https://github.com/Mungaramary/Phase3/assets/99483846/862eb8e8-c4d2-415d-a7c2-8c540a889beb)
As from above we can see that as long as the calls exceed 4 then the rate of churn highens showing more customers leaving the business.

### MODELLING
####  1) Gradient Boosting Classifier

gbm_model = GradientBoostingClassifier() 
gbm_model.fit(X_train_over,y_train_over) 
y_pred_gbm = gbm_model.predict(X_test)

Importance =pd.DataFrame({"Importance": gbm_model.feature_importances_*100},index = X_train_over.columns)
Importance.sort_values(by = "Importance", axis = 0, ascending = True).tail(15).plot(kind ="barh", color = "green",figsize=(9, 5))
plt.title("Feature Importance Levels");
plt.show()

It gives us the features as 

 ![random feature](https://github.com/Mungaramary/Phase3/assets/99483846/28bcbc7d-9716-4807-88cb-c147b0c20396)

 GRADIENT BOOSTING CLASSIFIER MODEL RESULTS 
Accuracy score for testing set:  0.9256
F1 score for testing set:  0.77903
Recall score for testing set:  0.8062
Precision score for testing set:  0.75362
![gradient boost 1](https://github.com/Mungaramary/Phase3/assets/99483846/18381e29-9f58-404a-8dc9-411c629b6524)

#### After trying out various models . we use :

##### An ROC curve (receiver operating characteristic curve) that is  a graph showing the performance of a classification model at all classification thresholds. The curve plots two parameters: True Positive Rate. False Positive Rate

from sklearn.metrics import plot_roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

classifiers = [LogisticRegression(),

               RandomForestClassifier(),
               
               DecisionTreeClassifier(),
               
               GradientBoostingClassifier()]

Create an empty list to store the results

result_table = []

Train the models and record the results

for cls in classifiers:
   
model = cls.fit(X_train_over, y_train_over)

result_table.append((cls.__class__.__name__, model))

Plot the ROC curves

fig = plt.figure(figsize=(8,6))

for clf_name, model in result_table:

plot_roc_curve(model, X_test, y_test, name=clf_name, ax=plt.gca(), lw=2)

plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

plt.xlabel("False Positive Rate", fontsize=15)

plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)

plt.legend(prop={'size': 13}, loc='lower right')

plt.show()

![roc](https://github.com/Mungaramary/Phase3/assets/99483846/aec1e842-c451-4357-ac84-d741f9cb253f)

The best performing models will have a curve that hugs the upper left of the graph, which is the the random forest classifier in this case

#### For the model comparisons we use the cross validation , accuracy , F1 score amd the pecision , Recall

#### Model comparisions 
For the accuracy we have 1) Gradient Boosting Classifier at 93
                         2) Random Classifeir at 92
                         3) Decision Tree Classifier at 88
                         4) linear logistic at 77
The best performing model will have the highest accuracy.
Of the four models tested, random forest classifier has the highest accuracy

#### Based on the Recall 
The best performing model will have the highest recall that gives the highest recall score .
#### Final model HYPERPARAMETER TUNED RANDOM FOREST MODEL

HYPERPARAMETER TUNED RANDOM FOREST MODEL RESULTS 
Accuracy score for testing set:  0.91929
F1 score for testing set:  0.75
Recall score for testing set:  0.74419
Precision score for testing set:  0.75591

Giging us the confusion matrix below
![random confusion final](https://github.com/Mungaramary/Phase3/assets/99483846/9f02bc64-4a7a-430d-bf05-974de271a661)

### RECOMENDATIONS

![image](https://github.com/Mungaramary/Phase3/assets/99483846/922e41fe-0e65-4df6-8939-a7eae5989900)

