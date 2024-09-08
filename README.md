# Customer-Churn-Analysis-

➢ Title
“Customer Churn Analysis in Telecom Industry Using 
Machine Learning and Data Visualization” 

Objectives
Identify Key Factors Influencing Churn: Analyze telecom customer data to pinpoint significant churn factors and understand customer behavior.
Predict Customer Churn: Develop a Random Forest model to accurately predict churn probabilities and aid in proactive retention strategies.
Visualize Churn Data: Create interactive dashboards in Power BI to present churn trends, demographic insights, and predictive analytics results.

Dataset
Source: Kaggle Telecom Customer Churn Dataset
Features: Includes demographic details, service subscriptions, payment histories, and churn statuses.

Scope:

Data Collection and Preparation: Extract and clean customer data from SQL Server. Prepare datasets for analysis and machine learning.
Data Transformation: Utilize Power BI for transforming data and creating meaningful visualizations.
Predictive Modeling: Apply machine learning algorithms to predict customer churn and evaluate model performance.
Visualization and Insights: Develop interactive dashboards and visualizations to represent the data insights and prediction results.

Project Target
Visualize & Analyze Customer Data:
Demographic
Geographic
Payment & Account Info
Services
Study Churner Profile & Identify Areas for Implementing Marketing Campaigns
Identify a Method to Predict Future Churners
Metrics Required
Total Customers
Total Churn & Churn Rate
New Joiners
STEP 1 - ETL Process in SQL Server
1. Install SQL Server Management Studio (SSMS):
Download from: SSMS Download
2. Creating Database:
Open SSMS and connect to your SQL Server.
Run the following query to create a database:
sql
Copy code
CREATE DATABASE db_Churn;
3. Import CSV into SQL Server Staging Table:
Right-click on the new database > Tasks > Import Data > Flat File Source.
Browse your CSV file and follow the wizard.
Set customerId as primary key and change BIT data type to VARCHAR(50).
4. Data Exploration:
Check distinct values:
sql
Copy code
SELECT Gender, COUNT(Gender) AS TotalCount,
    COUNT(Gender) * 1.0 / (SELECT COUNT(*) FROM stg_Churn) AS Percentage
FROM stg_Churn
GROUP BY Gender;
sql
Copy code
SELECT Contract, COUNT(Contract) AS TotalCount,
    COUNT(Contract) * 1.0 / (SELECT COUNT(*) FROM stg_Churn) AS Percentage
FROM stg_Churn
GROUP BY Contract;
Check nulls:
sql
Copy code
SELECT 
    SUM(CASE WHEN Customer_ID IS NULL THEN 1 ELSE 0 END) AS Customer_ID_Null_Count,
    SUM(CASE WHEN Gender IS NULL THEN 1 ELSE 0 END) AS Gender_Null_Count,
    -- (Repeat for other columns)
    SUM(CASE WHEN Churn_Reason IS NULL THEN 1 ELSE 0 END) AS Churn_Reason_Null_Count
FROM stg_Churn;
5. Clean and Insert Data into Production Table:
sql
Copy code
SELECT 
    Customer_ID,
    Gender,
    Age,
    Married,
    State,
    Number_of_Referrals,
    Tenure_in_Months,
    ISNULL(Value_Deal, 'None') AS Value_Deal,
    -- (Other columns)
    ISNULL(Churn_Reason , 'Others') AS Churn_Reason
INTO [db_Churn].[dbo].[prod_Churn]
FROM [db_Churn].[dbo].[stg_Churn];
6. Create Views for Power BI:
sql
Copy code
CREATE VIEW vw_ChurnData AS
    SELECT * FROM prod_Churn WHERE Customer_Status IN ('Churned', 'Stayed');
sql
Copy code
CREATE VIEW vw_JoinData AS
    SELECT * FROM prod_Churn WHERE Customer_Status = 'Joined';
STEP 2 - Power BI Transform
1. Add New Columns:
Churn Status:

plaintext
Copy code
Churn Status = IF([Customer_Status] = "Churned", 1, 0)
Monthly Charge Range:

plaintext
Copy code
Monthly Charge Range = 
IF([Monthly_Charge] < 20, "< 20", 
IF([Monthly_Charge] < 50, "20-50", 
IF([Monthly_Charge] < 100, "50-100", "> 100")))
2. Create Mapping Tables:
Age Group:

plaintext
Copy code
Age Group = IF([Age] < 20, "< 20", 
IF([Age] < 36, "20 - 35", 
IF([Age] < 51, "36 - 50", "> 50")))
plaintext
Copy code
AgeGrpSorting = 
IF([Age Group] = "< 20", 1, 
IF([Age Group] = "20 - 35", 2, 
IF([Age Group] = "36 - 50", 3, 4)))
Tenure Group:

plaintext
Copy code
Tenure Group = 
IF([Tenure_in_Months] < 6, "< 6 Months", 
IF([Tenure_in_Months] < 12, "6-12 Months", 
IF([Tenure_in_Months] < 18, "12-18 Months", 
IF([Tenure_in_Months] < 24, "18-24 Months", ">= 24 Months"))))
plaintext
Copy code
TenureGrpSorting = 
IF([Tenure Group] = "< 6 Months", 1, 
IF([Tenure Group] = "6-12 Months", 2, 
IF([Tenure Group] = "12-18 Months", 3, 
IF([Tenure Group] = "18-24 Months", 4, 5))))
3. Create Table for Services:
Unpivot services columns:

From: Attribute to Services, Value to Status.
STEP 3 - Power BI Measure
Total Customers:

plaintext
Copy code
Total Customers = COUNT(prod_Churn[Customer_ID])
New Joiners:

plaintext
Copy code
New Joiners = CALCULATE(COUNT(prod_Churn[Customer_ID]), prod_Churn[Customer_Status] = "Joined")
Total Churn:

plaintext
Copy code
Total Churn = SUM(prod_Churn[Churn Status])
Churn Rate:

plaintext
Copy code
Churn Rate = [Total Churn] / [Total Customers]
STEP 4 - Power BI Visualization
1. Summary Page:
Top Card:

Total Customers
New Joiners
Total Churn
Churn Rate %
Demographic:

Gender – Churn Rate
Age Group – Total Customer & Churn Rate
Account Info:

Payment Method – Churn Rate
Contract – Churn Rate
Tenure Group – Total Customer & Churn Rate
Geographic:

Top 5 State – Churn Rate
Churn Distribution:

Churn Category – Total Churn
Tooltip: Churn Reason – Total Churn
Service Used:

Internet Type – Churn Rate
Services – Status – % RT Sum of Churn Status
2. Churn Reason Page (Tooltip):
Churn Reason – Total Churn
STEP 5 – Predict Customer Churn
1. Data Preparation for ML Model:
Import Views into Excel:

Go to Data >> Get Data >> SQL Server Database.
Import vw_ChurnData & vw_JoinData.
Save the file as Prediction_Data.
2. Create Churn Prediction Model – Random Forest:
Install Anaconda and Required Libraries:

Install Anaconda from: Anaconda Installation
Run the following commands:
bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn joblib
Jupyter Notebook Code:

python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
file_path = r"C:\yourpath\Prediction_Data.xlsx"
data = pd.read_excel(file_path, sheet_name='vw_ChurnData')

# Data preprocessing
data = data.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)
columns_to_encode = [
    'Gender', 'Married', 'State', 'Value_Deal', 'Phone_Service', 'Multiple_Lines',
    'Internet_Service', 'Internet_Type', 'Online_Security', 'Online_Backup',
    'Device_Protection_Plan', 'Premium_Support', 'Streaming_TV', 'Streaming_Movies',
    'Streaming_Music', 'Unlimited_Data', 'Contract', 'Paperless_Billing',
    'Payment_Method'
]
label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])
data['Customer_Status'] = data['Customer_Status'].map({'Stayed': 0, 'Churned': 1})

# Split data
X = data.drop('Customer_Status', axis=1)
y = data['Customer_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(rf_model, 'churn_prediction_model.pkl')
Load the Model and Make Predictions:

python
Copy code
import joblib
model = joblib.load('churn_prediction_model.pkl')

# Sample data for prediction
sample_data = pd.DataFrame({
    # Add sample features here
})
predictions = model.predict(sample_data)
3. Evaluate Model Performance:
Analyze the confusion matrix and classification report to understand the performance metrics like precision, recall, F1 score, and accuracy.

