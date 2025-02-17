# AI Project - Fraud Detection in Credit Card Transactions

## Project Overview

This project aims to detect fraudulent credit card transactions from a list of transaction records. The dataset includes transaction details such as transaction time, amount, and personal or merchant details. The focus is on identifying patterns and features that could be indicative of fraudulent activities.

## Steps and Workflow

### Step 0: Goal Definition

The primary objective of this project is to detect fraudulent credit card transactions based on historical data. This will be achieved through data exploration, preprocessing, model training, and validation.

### Step 1: Data Acquisition
The dataset is loaded from a CSV file using pandas. This dataset contains records of credit card transactions with details such as transaction time, amount, and other features.

    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the dataset
    df_fraud = pd.read_csv('./credit_card_transactions.csv', sep=',')
    
    # Limit the dataset to the first 10,000 records for analysis
    df_small = df_fraud.head(10000)
    
    # Save the shortened dataset
    df_small.to_csv('credit_card_transactions_small.csv', index=False)
    
The dataset contains transaction details, including information about the time, amount, and personal/merchant details.

### Step 2: Data Exploration

**2.1 Initial Data Inspection**

We begin by inspecting the first few rows, data types, and basic statistics of the dataset.

    
    # Check the first 10 rows
    df_small.head(10)
    
    # Get general info and statistics about the dataset
    df_small.info()
    df_small.describe()
    
We observe that the dataset is imbalanced, with only 47 fraudulent transactions out of 10,000 records.

**2.2 Visualization**

Various visualizations are created to explore trends in the data.

    
    # Plot a scatter plot of fraud vs. transaction time
    df_small.plot(x='trans_date_trans_time', y='is_fraud', kind='scatter')
    plt.show()
    
    # Boxplot for merchant ZIP code
    df_small[["merch_zipcode"]].boxplot()
    

### Step 3: Data Preprocessing

**3.1 DateTime Conversion**

We convert the trans_date_trans_time and dob columns to datetime format for easier analysis and to extract useful insights (such as transaction time trends and age-related fraud patterns).

    
    df_small.loc[:, 'trans_date_trans_time'] = pd.to_datetime(df_small['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
    df_small.loc[:, 'dob'] = pd.to_datetime(df_small['dob'], format='%Y-%m-%d')
    
    # Calculate the age of the cardholder
    df_small.loc[:, 'age'] = df_small['trans_date_trans_time'].dt.year - df_small['dob'].dt.year
    
**3.2 Age Category Conversion**
We create an age category feature to group individuals based on their age.

    df_small.loc[(df_small['age'] >= 14) & (df_small['age'] < 21), 'age_category'] = 'Teen'
    df_small.loc[(df_small['age'] >= 21) & (df_small['age'] < 35), 'age_category'] = 'Young Adult'
    df_small.loc[(df_small['age'] >= 35) & (df_small['age'] < 55), 'age_category'] = 'Middle-aged'
    df_small.loc[df_small['age'] >= 55, 'age_category'] = 'Senior'
    
    # Check the result
    print(df_small[['age', 'age_category']].head())
    
**3.3 Visualizing Numerical Data**

We visualize the distribution of numerical features such as amt (transaction amount) and age.

    import seaborn as sns
    
    numerical_columns = ['amt', 'age']
    plt.figure(figsize=(14, 6))
    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(1, len(numerical_columns), i)
        sns.histplot(df_small[column], bins=30, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

### Step 4: Processing

**4.1 Missing Data**

We address missing ZIP codes based on the latitude and longitude values. This step ensures that no critical data is left out during the modeling phase.

**4.2 Feature Engineering**

Transaction Time Analysis: Investigate the time of day and the day of the week when fraudulent transactions most often occur.
Age Category Analysis: Analyze which age groups are more likely to commit fraud.

**4.3 Model Building**

We will implement multiple classification models to detect fraud. The models include:

- Logistic Regression
- Decision Trees
- Random Forest

The models will be compared using several evaluation metrics such as **F1-score, precision, and ROC curve**.

**4.4 Clustering**

We will also explore unsupervised learning techniques such as clustering to group similar transactions and identify anomalous behavior.

### Step 5: Model Validation

We will compare the performance of different models using metrics like F1-score, precision, and ROC curves. After validating on the reduced dataset, the models will be applied to the full dataset.

Additionally, we will analyze the dataset for specific years (e.g., 2019 and 2020) to detect any year-over-year trends.

Conclusion
This project demonstrates the process of detecting fraudulent credit card transactions using machine learning techniques, including classification and clustering. By analyzing various features of credit card transactions, we aim to build a robust fraud detection system that can handle imbalanced data and provide valuable insights into fraudulent activity.
