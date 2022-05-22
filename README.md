# Fetal Health Classification
by Oki Samila Rici

# Data Background
The dataset includes fetal health classifications determined by obstetricians using cardiotocography (CTG) equipment.

One of the causes of the high infant mortality rate is hypoxia experienced by the fetus. Fetal hypoxia is a condition in which there are low oxygen levels and increased levels of carbon dioxide in the fetal blood.This is actually avoidable, as CTG can be used to monitor the fetus's well-being.

CTG is a tool used to monitor the activity and heart rate of the fetus as well as uterine contractions while the baby is in the womb. Through this examination, doctors can evaluate a healthy fetus before and during childbirth. If there are changes or fetal distress conditions, the doctor can immediately provide help. Monitoring is recommended when the pregnancy enters the 3rd trimester or is more than 28 weeks pregnant.

CTG generally includes two small plates that are placed on the surface of the abdomen using an elastic belt that is wrapped around the pregnant woman's abdomen. One plate is used to measure the fetal heart rate, while the other plate is used to measure the strength and contractions of a pregnant woman's uterus.

Before CTG is used, the doctor will apply a special gel first to the pregnant woman's stomach. After that, the plates and belts from the CTG will be attached to the pregnant woman's stomach.

After a few minutes, the CTG dish connected to the CTG machine will display data on uterine contractions, fetal heart rate, and fetal activity in the uterus via the monitor screen. The data can also be printed on special paper that depicts the CTG graph.

Data source by Kaggle https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification.

# Questions
* What is the best model for predicting this data, and what is the this dataset percent accuracy?
* What variables can be used to categorize a fetus, and what is the relatioanship between these variables?

# Objective Statement
* Create a predictive model to know the normal fetus to prevent the adverse fetal outcome.
* Analyze the features that can be used to categorize the state of the fetus and find out the relationship between these features.

# Expected Outcome
* Obtain the best model from several machine learning models that can be used as a reference in determining the correctness of the dataset in predicting fetal well-being.
* Know about how a normal fetus can prevent fetal death.
* Know the variables that determine the category of the fetus and how the relationship between these variables.

# Data Dictionary
The dataset has 22 columns and 2126 rows:
  * baseline_FHR: basic heart rate when the uterus is at rest (per minute).
  * accelerations: increase in fetal heart rate (per second).
  * fetal_movement: calculation of baby kicks (per second).
  * uterine_contractions: fetal contractions to measure labor activity (per second).
  * light_decelerations: light decrease in heart rate (per second).
  * severe_decelerations: severe decrease in heart rate (per second).
  * prolongued_deceleratons: prolonged decrease in heart rate (per second).
  * percentage_STV: percentage of time the heart rate interval differs in the short-term variability.
  * mean_STV: the average value of the difference in heart rate intervals in short-term variability.
  * percentage_LTV: percentage of the time difference in heart rate interval long-term variability.
  * mean_LTV: the average value of the difference in heart rate intervals in long_term variability.
  * histogram_width : width of FHR histogram
  * histogram_min : minimum of FHR histogram
  * histogram_max : maximum of FHR histogram
  * histogram_number_of_peaks : number of histogram peaks
  * histogram_number_of_zeros : number of histogram zeros
  * histogram_mode : histogram mode
  * histogram_mean : histogram mean
  * histogram_median : histogram median
  * histogram_variance : histogram variance
  * histogram_tendency : histogram tendency
  * fetal_health : target variable that has 3 classes, namely normal, suspect and pathological

# Load Dataset
* The dataset contains 22 columns and 2126 rows. There are ten features histogram that doesn't have a clear explanation from the source and also don't know a relationship with the target variable. Thus, the columns will be deleted. The analysis will focus on uterine contractions, fetal movement, and an increase and decrease in fetal heart rate. Some of the columns will be renamed to make them easier to understand.
* After the histogram column is removed, there are 12 columns and 1 column is the target variable, namely fetal health. All columns are numeric with type float.

# Baseline Modal
The baseline model aims to compare with modeling after data cleansing. To assess the model, a benchmark is needed. A baseline model can serve as a benchmark, enabling a more informative evaluation of a modeling. A baseline model can also provide insight by showing which features it deems to be most significant.

![Screenshot (379)](https://user-images.githubusercontent.com/95860293/169691841-dc1f87fc-3658-4909-9e42-f41d41c503c3.png)

It can be seen that before cross-validation the F1 Score value for each model was high, reaching 94%. The F1 Score is very high, but not representative. The use of cross-validation parameters to get a more stable and accurate F1 Score value, because cross-validation divides the dataset into several fold so that the model can carry out learning in more detail.  So, the best model is Gradient-boosting with an F1 score of 80% using cross-validation.

Note: The explanation of the model selection can be found in the point the target variable check, and the metric used can be found in the point the boxplot to detect outliers.

# Data Cleansing
A. Missing Value : 0 missing value

Note:
The way to handle missing value

For numeric data:
* If the missing data is more than 60%, then drop the column.
* If less than 60% can be done drop the rows. In addition, it can fill in the mean value for data that has a normal distribution and if the data is skewed, it can fill in the median value.

For categorical data:
* If the missing data is more than 60%, then drop the column.
* If less than 60% can be done drop the rows. In addition, it can fill with the value of the mode, which is the value that occurs frequently. or it can be filled with a constant value by forming a new column, namely others.

The purpose of handling missing values is to make the data easier to analyze and the data to be more accurate as well as the machine learning model is to be made more powerful and without errors.

B. Duplicate Data : 14 rows of duplicate data.

Duplicate data can cause the model to misunderstand the data. The model will learn patterns that do not exist in reality or the model will study the same data as many duplicates. So it will produce a high accuracy value. We will assume a high accuracy value is good, but it is not. Therefore, before building a machine learning model we have to clean up duplicate data by deleting it. So that the modeling is more accurate.

# Data Understanding
A. Statistical Summary

![Screenshot (390)](https://user-images.githubusercontent.com/95860293/169695836-4d882a8e-9b9b-4f2b-a1ec-ad5a25728f39.png)

* min and max values for each column look appropriate.
* mean > median in acceleration, fetal_movement, mild_deceleration, severe_deceleration, prolonged_deceleration, mean_STV, LTV_percentage, mean_LTV, and fetal health, indicating a positively skewed distribution.
* mean < median in uterine_contraction and STV_percentage, indicating a negatively skewed distribution. We will see this more clearly in the KDE plot distribution point.

B. Target Variabel Check

![Screenshot (391)](https://user-images.githubusercontent.com/95860293/169695923-2aad866f-4094-4a12-8377-5b925aa68198.png)

The count plot of the target variable indicates an imbalanced class. This means we cannot use accuracy as a metric to evaluate the performance of our model. The most appropriate metric for model evaluation can be:
  * AUC : classifer accuracy for imbalanced data
  * Precision : predicted positive rate
  * Recall : actual positive rate
  * F1 Score : the average of recall and precision

Among the 4 matrices to be used is the F1 score because it is to summarize precision and recall by taking the harmonic alignment of both. Therefore we can minimize the false-positive and the false-negative rate.

# Univariate Analysis
A. Boxplot to Detect Outliers

![Screenshot (395)](https://user-images.githubusercontent.com/95860293/169696034-b470f737-2a3a-4f2c-bb64-6ffd8a7e3f4b.png)

The Boxplot shows that the average of variables has many outliers. So that some models that will use are robust models against outliers are:
  * Random Forest
  * Desicion Tree
  * Gradient Boosting
  * XGBoost

B. KDE Plot for Knowing the Distribution

![Screenshot (392)](https://user-images.githubusercontent.com/95860293/169696120-9c1ddda5-7404-4ff3-b1a5-ceaada4573a3.png)

* baseline_FHR has the most symmetrical distribution
* uterine_contraction and percentage_STV variable are bimodal because they have two peaks. This means it has the 2 highest values in the each fetaures.
* Features that have a positively skewed are acceleration, fetal_movement, light_deceleration, severe_deceleration, prolongued_deceleration, mean_STV, percentage_LTV, and mean_LTV.
* fetal_health is a target variable that has 3 values. Dominated by a value of 1 or a normal fetus which gives a positive skewed indication.

# Bivariate Analysis

B. Exploratory Data Analysis
  1. Baseline Fetal Heart Rate

![Screenshot (401)](https://user-images.githubusercontent.com/95860293/169696371-9c6a6c42-252f-4193-b82c-a605d2a6995d.png)

Fetal Heart Rate baseline conditions for all fetal categories are normal. Because in the normal fetus of the fetal heart baseline range between 110-160 bpm. So, the suspect and pathological also range from 110 - 160 bpm, which means that it is included in the normal fetal condition.

As a result, the baseline fetal heart rate doesn't provide information in the data set for prediction because it is already in normal conditions for all fetuses. Thus, in suspect and pathological cases, it is not a physiological problem that leads to fetal compromise.

  2. Acelerations

![Screenshot (415)](https://user-images.githubusercontent.com/95860293/169698439-35aed7fb-867e-4d91-b987-29b51a8b6e44.png)

![Screenshot (416)](https://user-images.githubusercontent.com/95860293/169698458-0a2841ae-605d-48f5-a4e5-4111601efc3e.png)

The two charts above show that the median value in normal fetuses is much higher than the median value in suspect and pathological fetuses, which appears to be 0. Normal fetal acceleration ranges from 0 to 0.0019 per second, While the acceleration in suspected and pathological cases does not exceed 0.0050 per second. It can be concluded that a healthy fetus experiences acceleration.

  3. Fetal Movement

![Screenshot (417)](https://user-images.githubusercontent.com/95860293/169698553-30581951-a67e-4865-af74-0c8fc71394fb.png)

It can be seen that in all three cases, the majority of the fetal movements are in the range of 0-0.1, in the normal case with outliers extending to 0.5, for suspected till between 0.4 and 0.5 and for pathological the values remain just below 0.4, none beyond. So, based on the median value all categories of fetuses did not experience fetal movement.

  4. Uterine Contractions

![Screenshot (418)](https://user-images.githubusercontent.com/95860293/169698584-d65f84d0-2522-46f7-9908-0e96cb1b81ea.png)

![Screenshot (419)](https://user-images.githubusercontent.com/95860293/169698590-4ab2be69-1d8c-497f-acb1-dd130818d4e2.png)

It can be seen that a normal fetus experienced uterine contractions more frequently than suspect and pathological. Most of the fetuses in the suspect and pathological categories did not experience uterine contractions at all. So, it can be concluded that a healthy fetus experiences uterine contractions.

  5. Prolonged Decelerations

![Screenshot (420)](https://user-images.githubusercontent.com/95860293/169698642-b1e0a0ec-2b29-46d2-9d10-e083e9ff6f3e.png)

![Screenshot (421)](https://user-images.githubusercontent.com/95860293/169698645-6694feeb-8871-46da-a93b-dee319aaf18f.png)

The two charts above show that:
* Normal fetus: All fetuses except a few show 0 prolonged deceleration.
* Suspect: All fetuses here also show 0 prolonged decelerations.
* Pathological: pathological cases occur at a rate ranging from 0 to 0.005 per second.

As a result, a healthy fetus does not experience prolonged deceleration.

Based on several features analyzed, it appears that uterine contractions, accelerations, and prolonged decelerations are determinants of fetal classifications. These features clearly distinguish between normal, suspected, and pathological fetuses. Now let's see how the relationship or effect of prolonged acceleration and deceleration when uterine contractions occur.

6. Relationship Between Uterine Contractions and Accelerations

![Screenshot (422)](https://user-images.githubusercontent.com/95860293/169698714-16846a91-6349-4788-b5a5-7c5302052f91.png)

In the regression plot, we can see the difference between each type of fetal health:

* for normal type fetus (1): negative correlation between uterine contractions and acceleration. When uterine contractions decreases, acceleration increases.
* for suspected and pathological fetal types (2 and 3): positive correlation between uterine contractions and acceleration. When the uterine contractions increase, the acceleration also increases.

So, a healthy fetus is when uterine contractions decrease, and there is an acceleration or increase in fetal heart rate.

7. Relationship Between Uterine Contractions and Prolonged Decelerations

![Screenshot (423)](https://user-images.githubusercontent.com/95860293/169698759-e849eceb-199d-4693-b4cc-447ddf7468a2.png)

we can see that for all types of fetuses, there is a positive correlation between uterine contractions and prolonged deceleration. As uterine contractions increase, prolonged decelerations also increase.

It can be concluded that, when uterine contractions increase, there is an increase in prolonged decelerations or a decrease in the fetal heart rate. However, less than 0.003 per second. The difference with the suspect is that the uterine contractions are below normal fetus. While pathological, uterine contractions under normal conditions and prolonged exceed 0.003 per second.

# Modeling

![Screenshot (374)](https://user-images.githubusercontent.com/95860293/169698813-ecb0b123-064c-42ac-9a59-e46d1202f20a.png)

The best model after cleansing is the same as the Baseline, namely Gradient Boosting. Where the baseline F1 score is slightly higher because there are 14 duplicate data. The model will learn repeatedly on duplicate data so that the F1 Score value becomes slightly higher. The value F1 Score after cleansing is 78%. Furthermore, hyperparameter tuning will be performed for the best model to increase the F1 Score value.

# Hyperparameter Tuning

![Screenshot (377)](https://user-images.githubusercontent.com/95860293/169698852-002ee82e-82aa-4f29-a851-5f792620afae.png)

Hyperparameter tuning succeeded in increasing the value of F1 Score Gradient Boosting to 85%.

# Feature Importance

![Screenshot (425)](https://user-images.githubusercontent.com/95860293/169698942-38b1e0d3-d6bb-438c-9501-387a97ff5ddf.png)

![image](https://user-images.githubusercontent.com/95860293/169698951-d83ad880-86a3-47ea-b580-c9826c8870d5.png)

# Insight and Recommendations

Insights:

This dataset has the value F1 Score of Gradient Boosting of 85%. This shows that this dataset can be used as a guide for fetal well-being to avoid fetal death.

Variables that can be used to determine the condition of the fetus or fetal category in this dataset are uterine contractions, accelerations, and prolonged decelerations.

CTG results, in normal fetal conditions:
  * Basic fetal heart rate with a range of 110 - 160 beats per minute.
  * uterine contractions with a range of 0 - 0.015 per second.
  * acceleration (increased fetal heart rate) with a range of 0 - 0.019 per second.
  * prolonged decelerations (prolonged decrease in fetal heart rate) do not occur or if they occur no more than 0.003 per second.
  
The relationship between variables that can determine fetal category is in healthy fetuses when uterine contractions decrease, accelerations increase and there is no prolonged deceleration as well as otherwise.

Recommendations:

Cardiotocography examination is very important for pregnant women to avoid disturbances related to fetal hypoxia. Especially pregnancy accompanied by complications such as pre-eclampsia, rupture of membranes, pregnancy more than 40 weeks, diabetes, hypertension, asthma, thyroid, chronic infectious diseases and complications of other diseases.
