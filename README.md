
# EXNO:4-DS
# Name:T.KAVINAJAI
# Register no:212223100020
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd 
df=pd.read_csv('bmi.csv')
df
```
<img width="1324" height="568" alt="image" src="https://github.com/user-attachments/assets/f6afb117-2718-4558-80c7-cc8ef9ea13ad" />

```
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,RobustScaler,MaxAbsScaler,LabelEncoder
df2 = df.copy()
enc = StandardScaler()

df2[['new_height', 'new_weight']] = enc.fit_transform(df2[['Height', 'Weight']])

df2
```

```
df3 = df.copy()
enc = MinMaxScaler()

df3[['new_height', 'new_weight']] = enc.fit_transform(df3[['Height', 'Weight']])

df3
```
<img width="733" height="444" alt="image" src="https://github.com/user-attachments/assets/7b1e6902-ed5c-42e6-b927-b40ec5700042" />

```
df4= df.copy()
enc = RobustScaler()

df4[['new_height', 'new_weight']] = enc.fit_transform(df4[['Height', 'Weight']])

df4
```
<img width="768" height="449" alt="image" src="https://github.com/user-attachments/assets/369c814f-fbae-4961-a0a0-75a3c71c3a1f" />

```
df5= df.copy()
enc = RobustScaler()

df5[['new_height', 'new_weight']] = enc.fit_transform(df5[['Height', 'Weight']])

df5
```
<img width="844" height="460" alt="image" src="https://github.com/user-attachments/assets/46edbe76-32f3-4aab-95d9-8cce1b013111" />

```
df6= df.copy()
enc = Normalizer()

df6[['new_height', 'new_weight']] = enc.fit_transform(df6[['Height', 'Weight']])

df6
```
<img width="967" height="462" alt="image" src="https://github.com/user-attachments/assets/ce889d79-e459-4255-b933-b0b073412166" />


```
df7= df.copy()
enc = Normalizer()

df7[['new_height', 'new_weight']] = enc.fit_transform(df6[['Height', 'Weight']])

df7
```
<img width="1241" height="503" alt="image" src="https://github.com/user-attachments/assets/128e1a03-d921-445c-83d2-b6d2743b3e03" />

```
df=pd.read_csv("income(1) (1).csv")
df
```
<img width="1283" height="828" alt="image" src="https://github.com/user-attachments/assets/608034cc-5689-4cd1-9cf9-aa59223f231d" />

```
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
le = LabelEncoder()


for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])


X = df_encoded.drop("SalStat", axis=1)
y = df_encoded["SalStat"]

```

```
x
```
<img width="1285" height="463" alt="image" src="https://github.com/user-attachments/assets/3deb6c49-b1f9-42ba-9e74-be4b7fb5f13f" />

```
y
```
<img width="832" height="254" alt="image" src="https://github.com/user-attachments/assets/503ca76b-29de-4d48-bfa1-0d628d8190ec" />


```
from sklearn.feature_selection import SelectKBest, chi2
chi2_selector=SelectKBest(chi2,k=5)
chi2_selector.fit(X,y)

selected_features_chi2 = X.columns[chi2_selector.get_support()]
print("Selected features (Chi-Square):", list(selected_features_chi2))
mi_scores=pd.Series(chi2_selector.scores_, index=X.columns)
print(mi_scores.sort_values(ascending=False))

```

<img width="1105" height="294" alt="image" src="https://github.com/user-attachments/assets/127cfa67-4bd0-4235-a2ec-9fde43489339" />

```
from sklearn.feature_selection import f_classif
anova_selector=SelectKBest(f_classif,k=5)
anova_selector.fit(X,y)

selected_features_anova = X.columns[chi2_selector.get_support()]
print("Selected features (Anova F-test):", list(selected_features_anova))
mi_scores=pd.Series(anova_selector.scores_, index=X.columns)
print(mi_scores.sort_values(ascending=False))

```

<img width="1042" height="297" alt="image" src="https://github.com/user-attachments/assets/eaf99389-3cf1-46ac-8102-77c438cac059" />

```
from sklearn.feature_selection import mutual_info_classif
mi_selector=SelectKBest(mutual_info_classif,k=5)
mi_selector.fit(X,y)

selected_features_mi = X.columns[mi_selector.get_support()]
print("Selected features (Mutual Info):", list(selected_features_anova))
mi_scores=pd.Series(anova_selector.scores_, index=X.columns)
print(mi_scores.sort_values(ascending=False))

```
<img width="1202" height="308" alt="image" src="https://github.com/user-attachments/assets/699b31df-f4a6-4785-a561-a72fd5c532a3" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


model = LogisticRegression(max_iter=100)


rfe = RFE(model, n_features_to_select=5)


rfe.fit(X, y)


selected_features_rfe = X.columns[rfe.get_support()]
print("Selected features (RFE):", list(selected_features_rfe))

```
<img width="1167" height="903" alt="image" src="https://github.com/user-attachments/assets/d0dcd6d9-f8aa-4bdc-9a9b-60935484c0c7" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector


model = LogisticRegression(max_iter=100)


rfe = RFE(model, n_features_to_select=5)


rfe.fit(X, y)


selected_features_rfe = X.columns[rfe.get_support()]
print("Selected features (RFE):", list(selected_features_rfe))

```
<img width="1118" height="818" alt="image" src="https://github.com/user-attachments/assets/26c0f368-fd72-4f88-9e4c-81a8e3fe5901" />


```
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X,y)
importances=pd.Series(rf.feature_importances_, index=X.columns)
selected_features_rf = importances.sort_values(ascending=False).head(5).index
print(importances)
print("Top 5 features (Random Forest Importance):",list(selected_features_rf))
```
<img width="1017" height="249" alt="image" src="https://github.com/user-attachments/assets/3f4b5b82-da20-4826-94c7-5ded057ae760" />

```
from sklearn.linear_model import LassoCV
import numpy as np

lasso=LassoCV(cv=5).fit(X,y)
importance=np.abs(lasso.coef_)

selected_features_lasso = X.columns[importance>0]
print("Selected features (Lasso):", list(selected_features_lasso))
```
<img width="1098" height="207" alt="image" src="https://github.com/user-attachments/assets/a27340a1-8a6f-4700-88d6-33c073c334b7" />

# RESULT:
Performing feature scaling and feature selection processes and succcessfullt saving the data to a file.
