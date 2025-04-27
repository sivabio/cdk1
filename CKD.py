# Google Colab: Upload File
# from google.colab import files

# Upload CSV file
# uploaded = files.upload()

# Ensure the filename matches the dataset
import pandas as pd
import numpy as np
import pickle
df_data = pd.read_csv("C:/Users/sivak/Envs/dj/djprojects/cdk/kidney_disease.csv")
# print(df_data.shape)  # Should print
# print(df_data.head())  # Check the first few rows
df_data.drop('id',axis=1, inplace=True)
# df_data.describe()
df_data.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'anemia', 'class']
# df_data.head()
text_columns = ['packed_cell_volume','white_blood_cell_count','red_blood_cell_count']
# for i in text_columns:
# print(f"{i} : {df_data[i].dtype}")

def convert_text_to_numeric(df_data, column):
    df_data[column] = pd.to_numeric(df_data[column], errors='coerce')

for column in text_columns:
    convert_text_to_numeric(df_data, column)
#   print(f"{column} : {df_data[column].dtype}")
# missing = df_data.isnull().sum()
# missing[missing>0].sort_values(ascending=False).head(26)
def mean_value_imputation(df_data, column):
    mean_value = df_data[column].mean()
 df_data[column].fillna(value=mean_value, inplace=True)
#   df_data[column] = df_data[column].fillna(value=mean_value, inplace=True)

def mode_value_imputation(df_data, column):
    mode = df_data[column].mode()[0]
    df_data[column] = df_data[column].fillna(mode)
# df_data.columns
num_cols = [col for col in df_data.columns if df_data[col].dtype!='object']
for col_name in num_cols:
    mean_value_imputation(df_data, col_name)

cat_cols = [col for col in df_data.columns if df_data[col].dtype=='object']
for col_name in cat_cols:
    mode_value_imputation(df_data, col_name)

# missing = df_data.isnull().sum()
# missing[missing>0].sort_values(ascending=False).head(26)
# print(f"diabetes_mellitus :- {df_data['diabetes_mellitus'].unique()}")
# print(f"coronary_artery_disease :- {df_data['coronary_artery_disease'].unique()}")
# print(f"class :- {df_data['class'].unique()}")
df_data['diabetes_mellitus'] = df_data['diabetes_mellitus'].replace(to_replace = {' yes':'yes', '\tno':'no', '\tyes':'yes'})
df_data['coronary_artery_disease'] = df_data['coronary_artery_disease'].replace(to_replace = '\tno', value='no')
df_data['class'] = df_data['class'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})
# print(f"diabetes_mellitus :- {df_data['diabetes_mellitus'].unique()}")
# print(f"coronary_artery_disease :- {df_data['coronary_artery_disease'].unique()}")
# print(f"class :- {df_data['class'].unique()}")
df_data['class'] = df_data['class'].map({'ckd': 1, 'not ckd': 0})
df_data['red_blood_cells'] = df_data['red_blood_cells'].map({'normal': 1, 'abnormal': 0})
df_data['pus_cell'] = df_data['pus_cell'].map({'normal': 1, 'abnormal': 0})
df_data['pus_cell_clumps'] = df_data['pus_cell_clumps'].map({'present': 1, 'notpresent': 0})
df_data['bacteria'] = df_data['bacteria'].map({'present': 1, 'notpresent': 0})
df_data['hypertension'] = df_data['hypertension'].map({'yes': 1, 'no': 0})
df_data['diabetes_mellitus'] = df_data['diabetes_mellitus'].map({'yes': 1, 'no': 0})
df_data['coronary_artery_disease'] = df_data['coronary_artery_disease'].map({'yes': 1, 'no': 0})
df_data['appetite'] = df_data['appetite'].map({'good': 1, 'poor': 0})
df_data['peda_edema'] = df_data['peda_edema'].map({'yes': 1, 'no': 0})
df_data['anemia'] = df_data['anemia'].map({'yes': 1, 'no': 0})
df_data.head(5)
# target_corr = df_data.corr()['class'].abs().sort_values(ascending=False)[1:]
# target_corr
from sklearn.model_selection import train_test_split
X = df_data.drop("class", axis=1)
y = df_data["class"]

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=25)

# print(f"'X' shape: {X_train.shape}")
# print(f"'Xtest' shape: {X_test.shape}")
# from sklearn.tree import DecisionTreeClassifier

# dct = DecisionTreeClassifier()
# dct.fit(X_train, y_train)
# y_pred_dct = dct.predict(X_test)
# y_pred_dct
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# models = []
# models.append(('Naive Bayes',GaussianNB()))
# models.append(('KNN', KNeighborsClassifier(n_neighbors=8)))
# models.append(('RandomForestClassifier',RandomForestClassifier()))
# models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
# models.append(('SVM',SVC(kernel='linear')))
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
model = RandomForestClassifier()
model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# model.fit(X_train, y_train)
# print(confusion_matrix(y_test,y_pred))
# print('\n')
# print("accuracy: ",accuracy_score(y_test,y_pred))
# print('\n')
# print("precision: ",precision_score(y_test,y_pred))
# print('\n')
# print("recall: ",recall_score(y_test,y_pred))
# print('\n')
# print("f1score: ",f1_score(y_test,y_pred))
# print('\n')
# for name,
#   print(name, model)
#    print()
#    model.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    print(confusion_matrix(y_test,y_pred))
#    print('\n')
#    print("accuracy: ",accuracy_score(y_test,y_pred))
#    print('\n')
#    print("precision: ",precision_score(y_test,y_pred))
#    print('\n')
#    print("recall: ",recall_score(y_test,y_pred))
#    print('\n')
#    print("f1score: ",f1_score(y_test,y_pred))
#    print('\n')
# new_features = [[0,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1]]
# Predicted_CDK = model.predict(new_features)
# if Predicted_CDK == 1 :
#   print("Possiblity for CHronic Kidney Disease : yes CDK", Predicted_CDK)
# else :
#    print("Possiblity for CHronic Kidney Disease : no CDK", Predicted_CDK)
pickle.dump(model, open('model.pkl', 'wb'))
