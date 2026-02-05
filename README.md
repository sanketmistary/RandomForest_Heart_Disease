Heart Disease Prediction using Random Forest Classifier

Description : 

This project aims to build a machine learning model to predict the presence of heart disease based on various patient attributes. A Random Forest Classifier is used for this classification task, leveraging its ensemble learning capabilities to achieve robust predictions.

Data Source : 

The dataset used in this project is the 'heart-disease-dataset' from Kaggle. It can be accessed via the identifier johnsmith88/heart-disease-dataset. The dataset contains 1025 records, each with 14 features related to patient health, including a target variable indicating the presence or absence of heart disease.

Technologies Used : 

Python: The primary programming language for the project.
pandas: Used for data manipulation and analysis.
scikit-learn: A comprehensive machine learning library, including:
train_test_split: For splitting data into training and testing sets.
StandardScaler: For feature scaling.
RandomForestClassifier: The machine learning model used.
classification_report: For evaluating model performance.
kagglehub: Used for convenient download of the Kaggle dataset directly into the environment.
Methodology
Data Loading: The dataset was downloaded using kagglehub and subsequently loaded into a pandas DataFrame for initial exploration and processing.
Data Preparation: The features (X) were separated from the target variable (y). The dataset was then split into training and testing sets, with 80% of the data allocated for training and 20% for testing. A random_state=42 was used to ensure reproducibility of the split.
Feature Scaling: Numerical features in both the training and testing sets were scaled using StandardScaler. The scaler was fitted only on the training data to prevent data leakage, and then used to transform both sets.
Model Training: A RandomForestClassifier was initialized with n_estimators=4 (number of trees in the forest) and random_state=42 for consistent results. The model was then trained using the scaled training data (x_train, y_train).
Prediction: After training, the model made predictions (y_pred) on the scaled test set (x_test).
Evaluation: The performance of the trained model was evaluated using classification_report, which provides key metrics such as precision, recall, f1-score, and support for each class, as well as overall accuracy.

Results : 

The Random Forest Classifier achieved strong performance metrics on the test set:

Accuracy: 0.97
Precision:
Class 0 (No Heart Disease): 0.96
Class 1 (Heart Disease): 0.97
Recall:
Class 0 (No Heart Disease): 0.97
Class 1 (Heart Disease): 0.96
F1-score:
Class 0 (No Heart Disease): 0.97
Class 1 (Heart Disease): 0.97
These results indicate that the model is highly effective in distinguishing between patients with and without heart disease, with balanced performance across both classes.

How to Run : 

To execute this project and reproduce the results in Google Colab:

Open the Notebook: Access this notebook directly in Google Colab.
Run All Cells: Simply run all the cells sequentially from top to bottom. This will automatically:
Download the 'heart-disease-dataset' using kagglehub.
Preprocess the data (split, scale).
Train the RandomForestClassifier model.
Generate and display the classification report and other evaluation metrics.
Final Task
Subtask:
Present the generated README content to the user.

Summary:

Data Analysis Key Findings
The project successfully developed a machine learning model to predict heart disease using a Random Forest Classifier.
The dataset, sourced from Kaggle (johnsmith88/heart-disease-dataset), consists of 1025 rows and 14 columns, including features like 'age', 'sex', and 'thal', and a 'target' variable for heart disease presence.
The methodology involved a train-test split (80/20 ratio with random_state=42), feature scaling using StandardScaler, and model training with a RandomForestClassifier initialized with n_estimators=4 and random_state=42.
The model demonstrated strong performance on the test set, achieving an overall accuracy of 0.97.
Performance metrics were consistently high across both classes:
Precision: 0.96 for class 0 (no heart disease) and 0.97 for class 1 (heart disease).
Recall: 0.97 for class 0 and 0.96 for class 1.
F1-score: 0.97 for both classes.
A comprehensive README.md file was successfully generated, detailing the project title, description, data source, technologies used (Python, pandas, scikit-learn, kagglehub), methodology, results, and instructions on how to run the notebook in Google Colab.
