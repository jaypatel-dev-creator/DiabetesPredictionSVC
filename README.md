#  Diabetes Prediction using Support Vector Machine (SVM)

A machine learning project that predicts the likelihood of diabetes using clinical health parameters.  
The model is built with a strong focus on **medical relevance**, **proper evaluation**, and **industry-aligned ML practices**.

---

##  Model Training Overview

- **Algorithm:** Support Vector Machine (SVM)  
- **Kernel:** Linear kernel (chosen for better interpretability)  
- **Class Weighting:** Applied to handle mild class imbalance  
- **Data Handling:** Model trained strictly on training data to prevent data leakage  

---

## Model Evaluation

### Metrics Used
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### Performance Summary

| Metric | Value |
|------|------|
| Training Accuracy | 77% |
| Test Accuracy | 75.3% |
| Precision (Diabetic) | 0.63 |
| Recall (Diabetic) | 0.72 |
| F1-score (Diabetic) | 0.67 |

The close gap between training and testing accuracy indicates **good generalization** and **minimal overfitting**.

---

##  Confusion Matrix Interpretation

| Actual / Predicted | Non-Diabetic | Diabetic |
|------------------|--------------|----------|
| **Non-Diabetic** | 77 | 23 |
| **Diabetic** | 15 | 39 |

### Key Insights
- Most diabetic patients were correctly identified (**high recall**).  
- Some non-diabetic cases were predicted as diabetic, which is acceptable in **medical screening systems**.  
- Priority was intentionally given to **recall** to minimize false negatives (missed diabetic cases), which are more critical in healthcare applications.

---

##  Key Highlights

- Proper handling of class imbalance  
- No data leakage during training or evaluation  
- Clean and reproducible machine learning pipeline  
- Medically aligned evaluation strategy (recall-focused)  
- Well-documented and GitHub-ready structure  

---

## 🛠 Tech Stack

- Python  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  
- Google Colab  

---

##  Future Improvements

- Hyperparameter tuning using **GridSearchCV**  
- ROC–AUC curve analysis  
- Feature importance visualization  
- Model deployment using **Flask** or **Streamlit**  
- Converting the pipeline into a production-ready API  

---

