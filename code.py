import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train.csv')
dfx = pd.read_csv('test.csv')

df.apply(lambda x: sum(x.isnull()), axis=0)
df['Employment_Info_4'].value_counts()

### for replacing missing values
# df['Employment_Info_4'].fillna(int(1),inplace=True) # for strings
df['Employment_Info_4'].fillna(df['Employment_Info_4'].mean(), inplace=True)  # for integers
df['Insurance_History_5'].fillna(df['Insurance_History_5'].mean(), inplace=True)  # for integers
df['Employment_Info_6'].fillna(df['Employment_Info_6'].mean(), inplace=True)  # for integers
df['Family_Hist_2'].fillna(df['Family_Hist_2'].mean(), inplace=True)  # for integers
df['Family_Hist_3'].fillna(df['Family_Hist_3'].mean(), inplace=True)  # for integers
df['Family_Hist_4'].fillna(df['Family_Hist_4'].mean(), inplace=True)  # for integers
df['Family_Hist_5'].fillna(df['Family_Hist_5'].mean(), inplace=True)  # for integers
df['Medical_History_1'].fillna(df['Medical_History_1'].mean(), inplace=True)  # for integers
df['Medical_History_10'].fillna(df['Medical_History_10'].mean(), inplace=True)  # for integers
df['Medical_History_15'].fillna(df['Medical_History_15'].mean(), inplace=True)  # for integers
df['Medical_History_24'].fillna(df['Medical_History_24'].mean(), inplace=True)  # for integers
df['Medical_History_32'].fillna(df['Medical_History_32'].mean(), inplace=True)  # for integers
p_1 = df['Product_Info_2']
df = df.drop('Product_Info_2', 1)
train_en = LabelEncoder()
p_1 = train_en.fit_transform(p_1)

df.dropna(axis=1)  # drops all columns containing null values
##### low variance filter
from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=0.06999)
vt.fit(df)
feature_indices = vt.get_support(indices=True)
feature_names = [df.columns[idx] for idx, _ in enumerate(df.columns) if idx in feature_indices]
dframe = pd.DataFrame(data=df, columns=feature_names)
dframe['Product_Info_2'] = p_1
dframe = dframe.drop('Medical_History_10', axis=1)
dframe = dframe.drop('Product_Info_7', axis=1)

p_i2 = dfx['Product_Info_2']
dfx = dfx.drop('Product_Info_2', 1)

dfx['Employment_Info_4'].fillna(df['Employment_Info_4'].mean(), inplace=True)  # for integers
dfx['Insurance_History_5'].fillna(df['Insurance_History_5'].mean(), inplace=True)  # for integers
dfx['Employment_Info_6'].fillna(df['Employment_Info_6'].mean(), inplace=True)  # for integers
dfx['Family_Hist_2'].fillna(df['Family_Hist_2'].mean(), inplace=True)  # for integers
dfx['Family_Hist_3'].fillna(df['Family_Hist_3'].mean(), inplace=True)  # for integers
dfx['Family_Hist_4'].fillna(df['Family_Hist_4'].mean(), inplace=True)  # for integers
dfx['Family_Hist_5'].fillna(df['Family_Hist_5'].mean(), inplace=True)  # for integers
dfx['Medical_History_1'].fillna(df['Medical_History_1'].mean(), inplace=True)  # for integers
dfx['Medical_History_10'].fillna(df['Medical_History_10'].mean(), inplace=True)  # for integers
dfx['Medical_History_15'].fillna(df['Medical_History_15'].mean(), inplace=True)  # for integers
dfx['Medical_History_24'].fillna(df['Medical_History_24'].mean(), inplace=True)  # for integers
dfx['Medical_History_32'].fillna(df['Medical_History_32'].mean(), inplace=True)  # for integers
vt1 = VarianceThreshold(threshold=0.06999)
vt1.fit(dfx)
feature_indices1 = vt1.get_support(indices=True)
feature_names1 = [dfx.columns[idx] for idx, _ in enumerate(dfx.columns) if idx in feature_indices1]
dframe1 = pd.DataFrame(data=dfx, columns=feature_names1)
p_i2 = train_en.fit_transform(p_i2)
dframe1['Product_Info_2'] = p_i2
dframe1 = dframe1.drop('Medical_History_10', 1)
dframe1 = dframe1.drop('Product_Info_7', 1)
#################################################
y = dframe['Response']
x = dframe.drop('Response', 1)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=4500,min_samples_leaf=100,oob_score=True)
model.fit(x,y)
prediction = model.predict(dframe1)
print(prediction[0:40])
submission_dframe = pd.DataFrame()
submission_dframe['Id'] = dframe1['Id']
submission_dframe['Response'] = prediction
submission_dframe.to_csv('iyman.csv', index=False)
feature_list = list(x.columns)
feature_imp = pd.Series(model.feature_importances_, feature_list).sort_values(ascending=False)
print(feature_imp, feature_names == feature_names1)
# from sklearn import metrics
# print("Accuracy is :", metrics.accuracy_score(y,prediction))
