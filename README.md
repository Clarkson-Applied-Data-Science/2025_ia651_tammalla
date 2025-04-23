# Parkinson's Disease Detection Using Machine Learning

## Abstract
Parkinson's Disease detection is a major challenge in healthcare, requiring accurate and early detection methods. This project applies supervised machine learning techniques—Logistic Regression, Support Vector Classification (SVC), Decision Trees, Random Forest, and Neural Networks (MLP)—to detect Parkinson's Disease from voice features. The methodology includes data preprocessing, feature scaling, model training, hyperparameter tuning, and performance benchmarking based on accuracy, precision, recall, and F1-score.

## Dataset

- **Source**: [Parkinson's Disease Voice Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Total Records**: 197
- **Attributes**: Various voice-related features such as `HNR`, `RPDE`, `MDVP`, `PPE`, `Jitter` total of 22 variables.

## Methodology

### Preprocessing
- **Normalization**: Applied **StandardScaler** to standardize the features and ensure they are on the same scale for better model performance.
- **Class Balancing**: Used **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance in the dataset, ensuring that both classes (Healthy vs Parkinson's) are adequately represented in the training set.
- **Feature Selection**: Removed highly correlated features based on Pearson correlation, reducing multicollinearity and enhancing model performance.
- **Feature Scaling**: After balancing, the scaled features were used to train the models, ensuring uniformity in training.

### Exploratory Data Analysis
- **Analyzed feature distributions**: Visualized the distribution of features to understand how the data is spread and whether there are any skewed distributions.
- **Correlation Analysis**: Performed a correlation analysis to identify and remove redundant features, ensuring the models focus on the most informative features.
- **Class Distribution**: Evaluated the distribution of labels to confirm the need for balancing due to the underrepresentation of the Parkinson's class.

### Model Training
- **Logistic Regression**: Used as a baseline model to assess linear relationships between the features and the target variable.
- **Support Vector Classification (SVC)**: Applied with a radial basis function kernel, tuned for better performance using **GridSearchCV**.
- **Decision Tree**: Used for non-linear classification and as a benchmark for other tree-based models.
- **Random Forest**: Employed an ensemble of decision trees, trained with hyperparameter tuning using **GridSearchCV**.
- **Neural Networks (MLP)**: A deep learning model used for its ability to learn complex patterns from the data, trained with **GridSearchCV**.





### Model Evaluation
- Evaluated each model using metrics like **accuracy**, **precision**, **recall**, and **F1-score** to ensure that both performance and the balance between false positives and false negatives were considered.
- Plotted **ROC curves** for each model to assess how well the models perform at different classification thresholds.

## Model Evaluation

| Model              | Accuracy  | Precision | Recall   | F1-Score |
|--------------------|-----------|-----------|----------|----------|
| Logistic Regression | 0.743590  | 0.913043  | 0.724138 | 0.812511 |
| SVC                | 0.871795  | 0.961538  | 0.862069 | 0.909091 |
| Decision Tree      | 0.897436  | 0.962963  | 0.896552 | 0.929432 |
| Random Forest      | 0.923077  | 0.964286  | 0.931034 | 0.947368 |
| Neural Network     | 0.923077  | 0.964286  | 0.931034 | 0.947368 |


### Hyperparameter Tuning
- **GridSearchCV** was applied for **Random Forest** and **Neural Networks (MLP)** to optimize the models' hyperparameters, ensuring the best performance for both models.

## Model Evaluation After Hyperparameter Tuning

| Model              | Accuracy  | Precision | Recall   | F1-Score |
|--------------------|-----------|-----------|----------|----------|
| Random Forest (Tuned) | 0.923077  | 0.964286  | 0.931034 | 0.947368 |
| Neural Network (Tuned) | 0.923077  | 1.000000  | 0.896552 | 0.945455 |

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
