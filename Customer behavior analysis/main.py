
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


df = pd.read_csv('Ecommerce_Customer_Behavior.csv')

print(df.head())
print(df.info())

print(df.isnull().sum())

print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

sns.countplot(x='Satisfaction Level', data=df)
plt.title('Distribution of Satisfaction Level')
plt.show()

df['Satisfaction Level'] = df['Satisfaction Level'].fillna('Unknown')

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['City'] = label_encoder.fit_transform(df['City'])
df['Membership Type'] = label_encoder.fit_transform(df['Membership Type'])
df['Satisfaction Level'] = label_encoder.fit_transform(df['Satisfaction Level'])
print(df.head())

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

X = df.drop(columns=['Total Spend'])
y = df['Total Spend']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


decision_tree = DecisionTreeRegressor(random_state=42)
random_forest = RandomForestRegressor(random_state=42)
svm = SVR()

decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
svm.fit(X_train, y_train)

y_pred_dt = decision_tree.predict(X_test)
y_pred_rf = random_forest.predict(X_test)
y_pred_svm = svm.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)

print(f"Decision Tree - MSE: {mse_dt}, R2: {r2_dt}")
print(f"Random Forest - MSE: {mse_rf}, R2: {r2_rf}")
print(f"SVM - MSE: {mse_svm}, R2: {r2_svm}")

final_model = random_forest
final_model.fit(X, y)
y_pred_final = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred_final)
final_r2 = r2_score(y_test, y_pred_final)

print(f"Final Model - MSE: {final_mse}, R2: {final_r2}")

# consumer scatterplot

np.random.seed(42)
n_samples = 100

df = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Age': np.random.randint(18, 70, n_samples),
    'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Miami'], n_samples),
    'Membership Type': np.random.choice(['Gold', 'Silver', 'Bronze'], n_samples),
    'Items Purchased': np.random.randint(5, 25, n_samples),
    'Average Rating': np.random.uniform(3.0, 5.0, n_samples),
    'Discount Applied': np.random.choice([True, False], n_samples),
    'Days Since Last Purchase': np.random.randint(1, 60, n_samples),
    'Satisfaction Level': np.random.choice(['Satisfied', 'Neutral', 'Unsatisfied'], n_samples),
})

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['City'] = label_encoder.fit_transform(df['City'])
df['Membership Type'] = label_encoder.fit_transform(df['Membership Type'])
df['Satisfaction Level'] = label_encoder.fit_transform(df['Satisfaction Level'])

df['Total Spend'] = np.random.uniform(500, 1500, n_samples)
df['Will Buy'] = df['Total Spend'].apply(lambda x: 1 if x > 800 else 0)


X = df.drop(columns=['Total Spend', 'Will Buy'])
y = df['Will Buy']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
df['Predicted Will Buy'] = model.predict(X)


plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Items Purchased'], c=df['Predicted Will Buy'], cmap='coolwarm', edgecolors='k', s=100)
plt.title('Customer Purchase Prediction: Will Buy vs Will Not Buy', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Items Purchased', fontsize=12)
plt.colorbar(label='Will Buy (1) / Will Not Buy (0)')
plt.show()


# random consumer barplot

np.random.seed(42)
n_samples = 100

df = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Age': np.random.randint(18, 70, n_samples),
    'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Miami'], n_samples),
    'Membership Type': np.random.choice(['Gold', 'Silver', 'Bronze'], n_samples),
    'Items Purchased': np.random.randint(5, 25, n_samples),
    'Average Rating': np.random.uniform(3.0, 5.0, n_samples),
    'Discount Applied': np.random.choice([True, False], n_samples),
    'Days Since Last Purchase': np.random.randint(1, 60, n_samples),
    'Satisfaction Level': np.random.choice(['Satisfied', 'Neutral', 'Unsatisfied'], n_samples),
})

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['City'] = label_encoder.fit_transform(df['City'])
df['Membership Type'] = label_encoder.fit_transform(df['Membership Type'])
df['Satisfaction Level'] = label_encoder.fit_transform(df['Satisfaction Level'])
df['Total Spend'] = np.random.uniform(500, 1500, n_samples)
df['Will Buy'] = df['Total Spend'].apply(lambda x: 1 if x > 800 else 0)

X = df.drop(columns=['Total Spend', 'Will Buy'])
y = df['Will Buy']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

df['Predicted Will Buy'] = model.predict(X)

plt.figure(figsize=(8, 6))
buy_counts = df['Predicted Will Buy'].value_counts()
plt.bar(buy_counts.index, buy_counts.values, color=['red', 'green'], tick_label=['Will Not Buy', 'Will Buy'])
plt.title('Prediction of Customer Purchase: Will Buy vs Will Not Buy', fontsize=14)
plt.xlabel('Customer Purchase Decision', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.show()


models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']
mse_scores = [918.72, 131.83, 84.49, 171854.11]
r2_scores = [0.9932, 0.9990, 0.9994, 0.2710]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(models, mse_scores, color=['blue', 'orange', 'red', 'purple'])
plt.title('Model Comparison - Mean Squared Error (MSE)')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.bar(models, r2_scores, color=['blue', 'orange', 'red', 'purple'])
plt.title('Model Comparison - R-squared')
plt.xlabel('Models')
plt.ylabel('R-squared')

plt.tight_layout()
plt.show()

feature_importance = random_forest.feature_importances_
features = X_train.columns

importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='Green')
plt.title('Feature Importance from Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.gca().invert_yaxis()
plt.show()

# Sample customer data
sample_customer = {
    'Gender': ['Female'],
    'Age': [30],
    'City': ['Los Angeles'],
    'Membership Type': ['Gold'],
    'Total Spend': [950.75],
    'Items Purchased': [12],
    'Average Rating': [4.3],
    'Discount Applied': [True],
    'Days Since Last Purchase': [20],
    'Satisfaction Level': ['Satisfied']
}

sample_df = pd.DataFrame(sample_customer)

training_data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Miami'],
    'Membership Type': ['Gold', 'Silver', 'Bronze', 'Gold', 'Silver'],
    'Satisfaction Level': ['Satisfied', 'Neutral', 'Unsatisfied', 'Satisfied', 'Neutral']
}

training_df = pd.DataFrame(training_data)

label_encoder_gender = LabelEncoder().fit(training_df['Gender'])
label_encoder_city = LabelEncoder().fit(training_df['City'])
label_encoder_membership = LabelEncoder().fit(training_df['Membership Type'])
label_encoder_satisfaction = LabelEncoder().fit(training_df['Satisfaction Level'])

sample_df['Gender'] = label_encoder_gender.transform(sample_df['Gender'])
sample_df['City'] = label_encoder_city.transform(sample_df['City'])
sample_df['Membership Type'] = label_encoder_membership.transform(sample_df['Membership Type'])
sample_df['Satisfaction Level'] = label_encoder_satisfaction.transform(sample_df['Satisfaction Level'])

X_train = pd.DataFrame({
    'Gender': [0, 1],
    'Age': [25, 40],
    'City': [0, 1],
    'Membership Type': [1, 0],
    'Total Spend': [800, 1200],
    'Items Purchased': [10, 15],
    'Average Rating': [4.0, 4.5],
    'Discount Applied': [1, 0],
    'Days Since Last Purchase': [30, 10],
    'Satisfaction Level': [2, 0]
})
y_train = [1, 1]

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

prediction = random_forest.predict(sample_df)

if prediction[0] == 1:
    result = "Will Buy"
else:
    result = "Will Not Buy"

print(f"Prediction for the sample customer: {result}")

data = {'Will Buy': 1 if result == "Will Buy" else 0, 'Will Not Buy': 1 if result == "Will Not Buy" else 0}
labels = [key for key, value in data.items() if value > 0]

plt.figure(figsize=(6, 6))
plt.pie([data[key] for key in labels], labels=labels, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
plt.title('Prediction Result: Will Buy vs Will Not Buy')
plt.show()
