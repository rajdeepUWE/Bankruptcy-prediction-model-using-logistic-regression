# Bankruptcy-prediction-model-using-logistic-regression

Open the .ipynb file for the complete code. The dataset "bank.csv" is taken from kaggle, Taiwan Economical Journal. 

Step 1-5: Data Loading and Visualization

The code starts by importing the necessary libraries: numpy, pandas, matplotlib.pyplot, and seaborn.
The dataset "bank.csv" is loaded using pd.read_csv() into a pandas DataFrame called data.
The first five rows of the dataset are displayed using data.head().
A heatmap of the correlation matrix of the dataset is plotted using sns.heatmap(data.corr()). This heatmap visualizes the correlations between different features in the dataset.
plt.show() is used to display the heatmap.
Step 6-8: Data Preprocessing and Splitting

The dataset is split into input features (X) and the target variable (y) using data.drop(["Bankrupt?"], axis="columns") and data["Bankrupt?"], respectively.
The train_test_split function is used to split the dataset into training and testing sets. The testing set size is set to 20% of the total dataset.
Step 9-11: Logistic Regression Model

A logistic regression model is instantiated using LogisticRegression() from scikit-learn.
The model is fitted to the training data using logreg.fit(x_train, y_train).
The accuracy score of the model on the test data is calculated using logreg.score(x_test, y_test).
Outcome: The logistic regression model achieves an accuracy of approximately 0.956 on the test data.

Step 12-29: Feature Importance Ranking

The code calculates the importance scores of features in the trained logistic regression model using logreg.coef_[0].
The indices of features are sorted in descending order based on their importance scores using np.argsort(feature_importances)[::-1].
The code then prints out the ranked list of features along with their importance scores.
Outcome: The code lists the features in descending order of their importance scores. However, these scores are extremely small, suggesting that the logistic regression model might not be capturing meaningful feature importance.

Step 30-55: Random Forest Classifier and F1 Score

A Random Forest classifier is instantiated using RandomForestClassifier().
The model is fitted to the training data, and predictions are made on the test data.
The F1 score, a metric that considers both precision and recall, is calculated using f1_score(y_test, rf_predictions).
Outcome: The Random Forest classifier achieves an F1 score of approximately 0.203 on the test data. A low F1 score indicates poor performance in identifying positive class instances (bankrupt companies) correctly.

Step 56-60: Creating a Pipeline with Standard Scaling and Logistic Regression

A data preprocessing pipeline is created using make_pipeline(StandardScaler(), LogisticRegression()).
The pipeline includes standard scaling of features and logistic regression.
Step 61-64: Training and Evaluating the Pipeline

The pipeline is trained on the training data using pipeline.fit(x_train, y_train).
The accuracy score of the pipeline on the test data is calculated using pipeline.score(x_test, y_test).
Outcome: The pipeline achieves an accuracy of approximately 0.959 on the test data. This accuracy is slightly better than that of the standalone logistic regression model.

Step 65-76: Cross-Validation for Logistic Regression

Cross-validation is performed using cross_val_score with 5 folds on the logistic regression model.
The average cross-validation score is calculated using np.mean(cv_scores).
Outcome: The average cross-validation score for the logistic regression model is approximately 0.960, indicating relatively stable performance across different folds.

Overall Assessment:

The logistic regression model achieved a decent accuracy on the test data, around 0.956, indicating reasonable predictive ability.
The Random Forest classifier did not perform well with a low F1 score of about 0.203, suggesting that it struggles with identifying bankrupt companies.
The pipeline, combining standard scaling and logistic regression, achieved an accuracy of about 0.959, slightly better than the standalone logistic regression model.
The cross-validation results suggest consistent performance of the logistic regression model across different folds.
The feature importance scores from the logistic regression model were extremely small, potentially indicating that the model may not be effectively capturing feature importance.
It's important to note that the evaluation metrics and outcomes should be considered in the context of the specific problem and dataset. It's also a good practice to further fine-tune the models and explore other algorithms for potentially better results.
