from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

app = Flask(__name__)


print("Current working directory:", os.getcwd())

#import os
#filepath = (r"")
#if not os.path.exists(filepath):
    #raise FileNotFoundError(f"File not found: {filepath}")
#data = pd.read_csv(filepath)



# Load dataset and train models
data = pd.read_csv('insurance.csv')  # Replace with your dataset file

# Encode categorical variables
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['smoker'] = label_encoder.fit_transform(data['smoker'])
data['region'] = label_encoder.fit_transform(data['region'])

corr = data.corr()
os.makedirs('./sampleImages', exist_ok=True)

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr,cmap='Blues',annot=True,fmt=".2f",ax=ax)
plt.title("Dependencies of Medical Charges")
plt.savefig('./sampleImages/Cor')
#plt.show()

#EDA

plt.figure(figsize=(10,7))
plt.title("Smoker vs Charges")
sns.barplot(x='smoker',y='charges',data=data)
plt.savefig('./sampleImages/SmokervsCharges')
#plt.show()

plt.figure(figsize=(15,9))
plt.title("Age vs Charges")
sns.barplot(x='age', y='charges', data=data)
plt.xlabel("Age")
plt.ylabel("Medical Charges")
plt.savefig('./sampleImages/AgevsCharges')
#plt.show()

plt.figure(figsize=(15,9))
plt.title("Age vs bmi")
sns.boxplot(x='age', y='bmi', data=data)
plt.xlabel("Age")
plt.ylabel("bmi")
plt.grid(True)
plt.savefig('./sampleImages/Agevsbmi')

#plt.show()

plt.figure(figsize=(12,7))
plt.title("Bmi vs Charges")
sns.scatterplot(x='bmi', y='charges', data=data, hue='sex')
plt.xlabel("Bmi")
plt.ylabel("Medical Charges")
plt.savefig('./sampleImages/BmivsCharges')
#plt.show()

X = data.drop(['charges','region'],axis=1).values
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor= KNeighborsRegressor()
regressor.fit(X_train_scaled,y_train)
knn_pred=regressor.predict(X_test_scaled)

svm_model = SVR(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

knn_r2= r2_score(y_test,knn_pred)

svm_r2 = r2_score(y_test, svm_predictions)

rf_r2 = r2_score(y_test, rf_predictions)

if knn_r2 > max(svm_r2, rf_r2):
    final_model = regressor
    scaler_used = False 
    predictions = knn_predictions
elif svm_r2 > rf_r2:
    final_model = svm_model
    scaler_used = True #whether SVM needs scaling or not
    predictions = svm_predictions
else:
    final_model = rf_model
    scaler_used = False 
    predictions = rf_predictions


# Calculate standard deviation for prediction range
#std_dev = np.std(predictions)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the user
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi= float(request.form['bmi'])
        smoker = request.form['smoker']
        children = int(request.form['children'])

        # Encode user input
        sex_encoded = 1 if sex == 'male' else 0
        smoker_encoded = 1 if smoker == 'yes' else 0

        # Prepare input for prediction
        input_data = pd.DataFrame([[age, sex_encoded, bmi, smoker_encoded, children]], 
                                  columns=['age', 'sex', 'bmi', 'smoker', 'children'])

        # Scale input if necessary
        if scaler_used:
            input_data_scaled = scaler.transform(input_data)
            predicted_cost = final_model.predict(input_data_scaled)[0]
        else:
            predicted_cost = final_model.predict(input_data)[0]


        # Return the result to the user
        return render_template('index.html', 
                               predicted_cost=f"₹{predicted_cost:.2f}",
                               age=age, sex=sex, bmi=bmi, smoker=smoker, children=children)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



