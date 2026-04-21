# Create a comprehensive ML workflow Python file

content = """# =========================================
# COMPLETE ML LAB WORKFLOW TEMPLATE (COLAB READY)
# Covers: Classification, Regression, Clustering, PCA
# =========================================

# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np

# ===============================
# 2. LOAD DATASET (CHOOSE ONE)
# ===============================

# Method 1: Direct URL
# df = pd.read_csv("https://raw.githubusercontent.com/rolandmueller/titanic/main/titanic3.csv")

# Method 2: Local file
# df = pd.read_csv("data.csv")

# ===============================
# 3. INSPECT DATA
# ===============================
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()

# ===============================
# 4. DATA CLEANING
# ===============================

# Select relevant columns
# df = df[['col1','col2','target']]

# Handle missing values
# df['col'] = df['col'].fillna(df['col'].mean())
# df = df.dropna()

# Encode categorical variables
# df['col'] = df['col'].map({'A':0, 'B':1})

# ===============================
# 5. SPLIT FEATURES & TARGET
# ===============================
# X = df.drop("target", axis=1)
# y = df["target"]

# ===============================
# 6. TRAIN TEST SPLIT
# ===============================
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ===============================
# 7. SCALING (IF REQUIRED)
# ===============================
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# ===============================
# -------- CLASSIFICATION --------
# ===============================

# Logistic Regression
from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# KNN
from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=5)

# SVM
from sklearn.svm import SVC
# model = SVC(kernel='rbf')

# Random Forest
from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
# model = AdaBoostClassifier(n_estimators=50)

# ===============================
# EVALUATION (CLASSIFICATION)
# ===============================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# ===============================
# -------- REGRESSION --------
# ===============================

from sklearn.linear_model import LinearRegression
# model = LinearRegression()

from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor()

from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()

# ===============================
# EVALUATION (REGRESSION)
# ===============================
from sklearn.metrics import mean_squared_error, r2_score

# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# ===============================
# -------- CLUSTERING --------
# ===============================
from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# labels = kmeans.predict(X)

# ===============================
# -------- PCA --------
# ===============================
from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X)

# ===============================
# END OF TEMPLATE
# ===============================
"""

file_path = "/mnt/data/ml_lab_full_template.py"
with open(file_path, "w") as f:
    f.write(content)

file_path
