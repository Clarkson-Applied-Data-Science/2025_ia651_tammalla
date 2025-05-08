# Parkinson's Disease Detection Using Machine Learning

## Abstract
Parkinson's Disease detection is a major challenge in healthcare, requiring accurate and early detection methods. This project applies supervised machine learning techniques—Logistic Regression, Support Vector Classification (SVC), Decision Trees, Random Forest, and Neural Networks (MLP)—to detect Parkinson's Disease from voice features. The methodology includes data preprocessing, feature scaling, model training, hyperparameter tuning, and performance benchmarking based on accuracy, precision, recall, and F1-score.

## Dataset

- **Source**: [Parkinson's Disease Voice Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Total Records**: 197
- **Attributes**: Various voice-related features such as `HNR`, `RPDE`, `MDVP`, `PPE`, `Jitter` total of 22 variables.

## Summary of all variables

| Feature              | Mean     | Std Dev  | Min     | 25%      | Median   | 75%      | Max      |
|----------------------|----------|----------|---------|----------|----------|----------|----------|
| MDVP:Fo(Hz)          | 154.23   | 41.39    | 88.33   | 117.57   | 148.79   | 182.77   | 260.11   |
| MDVP:Fhi(Hz)         | 197.10   | 91.49    | 102.15  | 134.86   | 175.83   | 224.21   | 592.03   |
| MDVP:Flo(Hz)         | 116.32   | 43.52    | 65.48   | 84.29    | 104.32   | 140.02   | 239.17   |
| MDVP:Jitter(%)       | 0.00622  | 0.00485  | 0.00168 | 0.00346  | 0.00494  | 0.00737  | 0.03316  |
| MDVP:Jitter(Abs)     | 0.00004  | 0.00004  | 0.00001 | 0.00002  | 0.00003  | 0.00004  | 0.00026  |
| MDVP:RAP             | 0.00031  | 0.00030  | 0.00068 | 0.00017  | 0.00025  | 0.00038  | 0.00214  |
| MDVP:PPQ             | 0.00034  | 0.00028  | 0.00092 | 0.00019  | 0.00027  | 0.00040  | 0.01958  |
| Jitter:DDP           | 0.00094  | 0.00089  | 0.00204 | 0.00049  | 0.00075  | 0.00115  | 0.06433  |
| MDVP:Shimmer         | 0.02927  | 0.01886  | 0.00954 | 0.01651  | 0.02297  | 0.03789  | 0.11908  |
| MDVP:Shimmer(dB)     | 0.28225  | 0.19488  | 0.08500 | 0.14850  | 0.22210  | 0.35750  | 1.30200  |
| Shimmer:APQ3         | 0.01566  | 0.01015  | 0.00455 | 0.00825  | 0.01279  | 0.02287  | 0.05647  |
| Shimmer:APQ5         | 0.01788  | 0.01020  | 0.00575 | 0.00958  | 0.01347  | 0.02238  | 0.07940  |
| MDVP:APQ             | 0.02079  | 0.01695  | 0.00719 | 0.01308  | 0.01826  | 0.02940  | 0.13778  |
| Shimmer:DDA          | 0.04699  | 0.03046  | 0.01364 | 0.02474  | 0.03836  | 0.06080  | 0.16942  |
| NHR                  | 0.02489  | 0.03045  | 0.00136 | 0.00930  | 0.01166  | 0.02578  | 0.31482  |
| HNR                  | 21.89    | 4.43     | 8.40    | 19.11    | 22.02    | 25.03    | 33.05    |
| RPDE                 | 0.49854  | 0.10394  | 0.24160 | 0.42131  | 0.42955  | 0.58756  | 0.68510  |
| DFA                  | 0.71810  | 0.05534  | 0.57428 | 0.67476  | 0.72225  | 0.76188  | 0.82529  |
| spread1              | -5.6844  | 1.09021  | -7.9649 | -6.4510  | -5.7209  | -5.0462  | -2.4340  |
| spread2              | 0.22652  | 0.08334  | 0.00627 | 0.17435  | 0.21889  | 0.27932  | 0.45050  |
| D2                   | 2.38183  | 0.38280  | 1.42329 | 2.09913  | 2.36153  | 2.63646  | 3.67115  |
| PPE                  | 0.02655  | 0.09012  | 0.04454 | 0.13745  | 0.19405  | 0.25298  | 0.52737  |

## Exploratory Data Analysis

# Class Distribution

![ClassDist](https://github.com/user-attachments/assets/b3774140-ae9b-4e4a-8648-11a285369f84)

The class distribution is imbalanced, with significantly more patients diagnosed with Parkinson’s (label 1) than healthy individuals (label 0). This imbalance may affect model performance and requires techniques like resampling or class weighting during training.

# Histogram of all features

![Histogram](https://github.com/user-attachments/assets/4e2a9deb-90e4-457d-bfba-2d619232e75e)

The histograms show that most features are right-skewed with a few outliers, especially in jitter, shimmer, and noise-related metrics. Some features like spread1, spread2, DFA, and D2 appear more normally distributed. This variability suggests that scaling and normalization are important before applying machine learning models.

# Box Plots

![Boxplot-1](https://github.com/user-attachments/assets/98e54741-c245-433f-b3b6-84401e66e918)

Jitter is also higher in Parkinson’s patients, indicating increased pitch variability and vocal irregularity compared to healthy individuals.

![Bookplot2](https://github.com/user-attachments/assets/e4eb555b-91eb-4fb3-86fe-3ea9e091df18)

Patients with Parkinson’s show significantly higher shimmer values, indicating greater amplitude variability in their voice. This reflects reduced vocal stability.

![boxplot3](https://github.com/user-attachments/assets/8d56f053-b275-4b66-8178-803f16a84985)

Healthy individuals tend to have higher HNR values, suggesting cleaner, less noisy vocal signals, while Parkinson’s patients have more hoarseness or breathiness.

![boxplot4](https://github.com/user-attachments/assets/99cdb9ce-6fa9-41b4-ac13-2ef8b0d2cf2d)

RPDE values are higher in those with Parkinson’s, suggesting more complex and less predictable voice patterns due to vocal impairment.

![boxplot5](https://github.com/user-attachments/assets/3271dc20-79df-459f-9545-3fd1af5b5eb0)

PPE is elevated in Parkinson’s cases, indicating increased vocal signal unpredictability, which is common in affected individuals.


# Stacked bar of 'RPDE', 'MDVP:Jitter(%)', 'PPE', 'HNR' vs Target variable

![Stacked bar plots](https://github.com/user-attachments/assets/a7d7cb6c-8d14-4ded-befc-3e81fcf259b2)

The stacked bar plots show how binned feature values relate to Parkinson’s status:

RPDE and PPE: Higher bins have more Parkinson’s cases, indicating increased voice signal complexity and unpredictability in affected individuals.

MDVP:Jitter(%): Most Parkinson’s cases are concentrated even in the lower jitter bin, showing even small pitch variations are informative.

HNR: Lower HNR bins have more Parkinson’s patients, reflecting noisier, less harmonic voices compared to healthy individuals.



### Methodology

## Preprocessing Steps:

This section outlines all the preprocessing steps performed from the train-test split onward, along with the specific dimensions at each stage.

1. **Train-Test Split**: The dataset was split into training and test sets using an 80-20 split.

   Training set shape: 156 rows, 22 features

   Test set shape: 40 rows, 22 features

2. **Feature Scaling**: StandardScaler was used to normalize the data by removing the mean and scaling to unit variance. The scaler was fit only on the training set and then applied to both training and test sets to prevent data leakage.

   Scaled training data shape: 156 rows, 22 features

   Scaled test data shape: 40 rows, 22 features

3. **Feature Reduction**: Removed highly correlated features based on a Pearson correlation matrix with a threshold of 0.9 to reduce multicollinearity and improve model interpretability.
Dropped features due to high correlation (> 0.9):
['MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'PPE']

![Co- relation matrix](https://github.com/user-attachments/assets/741c8adf-dc55-4f7d-8961-64aa0bb2bd2d)


5. **Class Balancing**: Used SMOTE (Synthetic Minority Over-sampling Technique) on the scaled training set to generate synthetic samples for the minority class and balance the dataset.
   
   **Note**: I have balanced two sets one with droping 10 columns and another without droping 
    10 columns to check if there is any difference in accuracy. 



### Model Training
- **Logistic Regression**: Used as a baseline model to assess linear relationships between the features and the target variable.
- **Support Vector Classification (SVC)**: Applied with a radial basis function kernel, tuned for better performance using **GridSearchCV**.
- **Decision Tree**: Used for non-linear classification and as a benchmark for other tree-based models.
- **Random Forest**: Employed an ensemble of decision trees, trained with hyperparameter tuning using **GridSearchCV**.
- **Neural Networks (MLP)**: A deep learning model used for its ability to learn complex patterns from the data, trained with **GridSearchCV**.

### Model Evaluation
- Evaluated each model using metrics like **accuracy**, **precision**, **recall**, and **F1-score** to ensure that both performance and the balance between false positives and false negatives were considered.
- Plotted **ROC curves** for each model to assess how well the models perform at different classification thresholds.

## Model Evaluation with droping 10 features

| Model              | Accuracy  | Precision | Recall   | F1-Score |
|--------------------|-----------|-----------|----------|----------|
| Logistic Regression | 0.743590  | 0.913043  | 0.724138 | 0.812511 |
| SVC                | 0.871795  | 0.961538  | 0.862069 | 0.909091 |
| Decision Tree      | 0.897436  | 0.962963  | 0.896552 | 0.929432 |
| Random Forest      | 0.923077  | 0.964286  | 0.931034 | 0.947368 |
| Neural Network     | 0.923077  | 0.964286  | 0.931034 | 0.947368 |

## Model Evaluation without droping 10 features.

| Model               | Accuracy | Precision (1) | Recall (1) | F1-score (1) |
|---------------------|----------|----------------|-------------|---------------|
| Logistic Regression | 0.769231 | 0.954545       | 0.724138    | 0.823529      |
| SVC                 | 0.820513 | 0.958333       | 0.793103    | 0.867925      |
| Decision Tree       | 0.846154 | 0.925926       | 0.862069    | 0.892857      |
| Random Forest       | 0.897436 | 0.962963       | 0.896552    | 0.928571      |
| Neural Network      | 0.974359 | 1.000000       | 0.965517    | 0.982456      |

 Interpretation: Clearly all models with feature reduction are having better accuracy a part from Neural Network which has 97 % accuracy and logistic regression which has 76 %.

## Confusion matrix for all models

# Logistic Regression

![LR](https://github.com/user-attachments/assets/777668d2-988f-4e1d-93ca-dac54ed6e54d)

# SVC

![SVC](https://github.com/user-attachments/assets/a6dc31b1-c073-464b-84ed-1e302b6fd0a7)


# Decision Tree

![DT](https://github.com/user-attachments/assets/9b18441d-f2e1-4d87-8e4b-37d9450764e7)

# Random Forest

![RF](https://github.com/user-attachments/assets/99907085-f799-4953-b9c1-68e022588829)

# Neural Network

![NN](https://github.com/user-attachments/assets/29193861-9ffd-4b8c-8d8c-d8cb2f02486f)

## ROC Curve

![ROC curve](https://github.com/user-attachments/assets/a18d75aa-3888-42af-a18a-066983c3999f)



### Hyperparameter Tuning
- **GridSearchCV** was applied for **Random Forest** and **Neural Networks (MLP)** to optimize the models' hyperparameters, ensuring the best performance for both models.

## Random Forest Classifier
The Random Forest model was optimized by performing a grid search over a predefined set of hyperparameters. This ensemble learning method constructs multiple decision trees during training and outputs the mode of their predictions.

# Tuned hyperparameters:
n_estimators: 50, 100, 150

max_depth: 5, 10, 15, None

min_samples_split: 2, 5

min_samples_leaf: 1, 2, 4

## Neural Network (MLP)
The Multi-layer Perceptron (MLP) model was also tuned using GridSearchCV. MLP is a type of feedforward neural network that learns non-linear relationships between input features and the target variable through multiple layers of neurons.

# Tuned hyperparameters:
hidden_layer_sizes: (50,), (100,), (50, 50)

activation: relu, tanh

solver: adam, sgd

alpha: 0.0001, 0.001, 0.01

## Model Evaluation After Hyperparameter Tuning

| Model              | Accuracy  | Precision | Recall   | F1-Score |
|--------------------|-----------|-----------|----------|----------|
| Random Forest (Tuned) | 0.923077  | 0.964286  | 0.931034 | 0.947368 |
| Neural Network (Tuned) | 0.923077  | 1.000000  | 0.896552 | 0.945455 |

## Feature Importance with Random Forest.

![Feature Importance](https://github.com/user-attachments/assets/fe04c6b3-c870-4fc0-b21f-197869bcddcf)

# Interpretation:
The Random Forest model identified spread1, MDVP:Shimmer, and MDVP:Fo(Hz) as the most important features in detecting Parkinson’s Disease. These features reflect voice instability and frequency variation, which are commonly affected in Parkinson’s patients. Lower-ranked features like DFA and HNR contributed less but still added value to the model's predictions.


## Conclusion

This project aimed to detect **Parkinson's Disease** from voice features using multiple machine learning models. The dataset was processed with careful preprocessing steps including feature scaling, class balancing using **SMOTE**, and feature selection based on correlation. Hyperparameter tuning was applied to **Random Forest** and **Neural Network (MLP)** using **GridSearchCV** to optimize model performance.

### Key Findings:
- **Random Forest** and **Neural Network (MLP)** both performed well with **similar accuracy** of approximately **92%**. 
- **Neural Network (MLP)** demonstrated **perfect precision** (1.0) but had slightly lower recall and F1 score compared to **Random Forest**.
- **Random Forest** showed a more balanced trade-off between **precision** and **recall**, resulting in a slightly higher **F1 score** (0.947) compared to the **Neural Network's F1 score** (0.945).

### Model Performance:
- **Random Forest** and **Neural Network (MLP)** both exhibited strong performance and were the top contenders.
- The **tuned Random Forest model** achieved the best balance between **precision** and **recall**, making it the preferred model for this task.

In conclusion, the results suggest that **Random Forest** is slightly better at handling the Parkinson's Disease classification task, especially when considering both false positives and false negatives. However, the **Neural Network (MLP)** also performed extremely well, particularly in precision, making it a good alternative depending on the specific needs of the application.

## Future Work

1. **Model Deployment**:
   - Deploy the best-performing model (**Random Forest** or **Neural Network**) to a web application using frameworks like **Flask** or **Streamlit**. This will enable real-time Parkinson's Disease prediction using voice features, providing an accessible tool for healthcare professionals or researchers.
   - Implement **API endpoints** for easy integration with other medical applications or systems.

2. **Exploring Other Models**:
   - Explore **ensemble models** like **XGBoost**, **LightGBM**, or **CatBoost**, which often outperform traditional Random Forest and Neural Networks in terms of predictive accuracy and efficiency, especially on imbalanced datasets.
   - Experiment with **Deep Learning models** such as **Convolutional Neural Networks (CNN)** or **Recurrent Neural Networks (RNN)** for more advanced feature extraction from time-series or sequential data.
   
3. **Feature Engineering and Selection**:
   - Further explore **domain-specific features** related to Parkinson's Disease and **time-series analysis**. Additional features, such as **voice pitch variations over time**, could provide more context to the model.
   - Experiment with **advanced feature selection techniques** like **Recursive Feature Elimination (RFE)** or **LASSO** to refine the feature set and reduce overfitting.

4. **Class Imbalance Handling**:
   - Although **SMOTE** has been applied, more advanced techniques for handling class imbalance like **NearMiss** or **ClusterCentroids** could be explored to improve model performance on the minority class without losing valuable data in the majority class.

5. **Integration with Medical Systems**:
    - Work towards integrating the final model into **electronic health record (EHR) systems** for seamless data collection and analysis. This would help healthcare providers use voice data for early-stage Parkinson’s detection, improving the speed and accuracy of diagnoses.


## Acknowledgment

This project was developed as part of the academic course **IA651: Applied Machine Learning** at **Clarkson University** under the supervision of **Prof. Michael Gilbert**.

**Authors**  
Rahul Tammalla (1010517)
