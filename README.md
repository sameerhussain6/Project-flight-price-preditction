# Project-flight-price-preditction

## Project Overview
This project aims to predict flight prices based on various factors such as departure time, arrival time, source, destination, and airline. By leveraging machine learning techniques, the model provides insights into pricing trends and helps users make informed decisions when booking flights.

## Features
- Data preprocessing and cleaning for structured analysis.
- Exploratory Data Analysis (EDA) to understand patterns and trends in flight prices.
- Implementation of machine learning models to predict flight prices.
- Model evaluation and optimization to ensure accuracy.

## Project Structure
1. **`Problem Definition and Data Loading:`**
The notebook begins by defining the problem: analyzing a flight booking dataset to gain insights into passenger behavior and potentially predict flight prices.
It describes the dataset, including its size and features.
Necessary libraries like pandas, NumPy, matplotlib, and seaborn are imported for data manipulation, analysis, and visualization.
2. **`Exploratory Data Analysis (EDA):`**
**Data Shape and Cleaning:**
The shape of the dataset **12 features** and **300153 instances** is examined using df.shape.
A function null_and_duplicates is defined and used to check for null values and duplicates in the dataset.
An unnecessary column ('Unnamed: 0') is dropped using df.drop.
Data Types and Categorical Variables:
Data types of each column are checked using df.info().
Categorical variables are identified and analyzed.
Frequency distributions and bar plots are created for categorical variables to understand their distribution.
3.**`Numerical Variables and Outlier Analysis:`**
Numerical variables are identified and analyzed.
Box plots are created to visualize the distribution of numerical variables and detect outliers.
Outliers in 'duration' and 'price' are identified and removed based on the interquartile range (IQR).
**Correlation Analysis:**
A heatmap is generated using sns.heatmap to visualize the correlation between numerical features. This helps identify relationships between variables.
4.**`Data Visualization:`**
Line plots are used to examine the relationship between 'airline', 'departure_time', 'stops', and 'days_left' with 'price'.
Count plots and bar plots are used to visualize the frequency and distribution of various features.
Insights are drawn from these visualizations to understand patterns and trends in the data.
-5. **`Data Preprocessing:`**
**Label Encoding:**
Categorical features are converted into numerical representations using Label Encoding from sklearn.preprocessing. This step is necessary for many machine learning algorithms.
-6. **`Feature Selection (Variance Inflation Factor - VIF):`**
**VIF**is used to detect multicollinearity among features.
Features with high VIF scores are identified, and the 'flight' feature is dropped to address multicollinearity.
- 7.**` Model Building and Evaluation:`**
**Data Splitting:**
The dataset is split into training and testing sets using train_test_split from sklearn.model_selection to evaluate model performance on unseen data.
**Data Scaling:**
Standard scaling is applied to numerical features using StandardScaler from sklearn.preprocessing. This ensures that features with different scales do not disproportionately influence the model.
Model Training and Evaluation:
Four different regression models are trained and evaluated:
**Linear Regression**
**Decision Tree**
**Random Forest**
**XGBoost**
For each model, the following steps are performed:
Model training using the training data.
Predictions on the testing data.
Evaluation using metrics like **R-squared (R2), mean absolute error (MAE), root mean squared error (RMSE), and mean absolute percentage error (MAPE).**
Visualization of actual vs. predicted values using scatter plots.
**Model Comparison and Selection:**
The performance of all four models is compared based on the evaluation metrics.
**Random Forest** and **XGBoost** are identified as the most promising models based on their high R2 scores and lower error values.

8.**` Conclusion:`**
The notebook summarizes the findings, highlighting the best-performing models and their strengths.
It emphasizes the importance of model selection, evaluation, and potential further improvements through hyperparameter tuning or feature engineering.

## Requirements
- Python 3.8+
- Jupyter Notebook
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Usage
1. Load the dataset into the notebook.
2. Follow the steps outlined in the notebook to preprocess the data and explore trends.
3. Train and evaluate machine learning models.
4. Use the model to predict flight prices for new inputs.

## Dataset
- Number of records -300153
- Features included -12

## Results
"In this analysis, we explored four different **machine learning** models to predict flight prices: **Linear Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**. We evaluated the models using key metrics such as **R-squared(R2)** and **RMSE**.

- **`Linear Regression:`** While providing a decent baseline with an R2 score of 0.91, it exhibited higher error values (RMSE: 6943, MAE: 4616, MAPE: 0.44) indicating limitations in capturing complex relationships within the data.
- **`Decision Tree:`** Descision Tree Was prone to overfitting, although it had good performance on the data,It achieved a R2 score of 0.98, indicating a better fit compared to Linear Regression. The error values were also considerably lower (RMSE: 3488, MAE: 1176, MAPE: 0.08)
- **`Random Forest:`** Random Forest showed significant improvement with the highest R2 score of 0.99, signifying a strong ability to handle complex relationships and reduce overfitting. This model also boasts the lowest error values among the four (RMSE: 2715, MAE: 1245, MAPE: 0.09).
- **`XGBoost:`** XGBoost While not as performant as Random Forest, XGBoost still delivered excellent results with an R2 score of 0.98 and relatively low error values (RMSE: 3469, MAE: 2002, MAPE: 0.15). This model showcases its strength in handling prediction tasks

- Based on the evaluation, the **Random Forest** and **XGBoost** models are the most promising for flight **price prediction**. Further hyperparameter tuning and feature engineering could potentially enhance the accuracy of these models even further.

- This analysis highlights the importance of **model selection** and **evaluation** in building accurate **machine learning** models for real-world applications. By carefully comparing performance and understanding the strengths and limitations of different algorithms, we can make informed decisions to create valuable predictive tools."

## Future Work
- Integration with real-time flight price APIs for dynamic predictions.
- Implementation of deep learning models for improved accuracy.
- Deployment of the model as a web application.

**`PROJECT BY-SHAIK MOHAMMAD SAMEER HUSSAIN`**
