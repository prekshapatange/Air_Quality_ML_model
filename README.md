# 🌍 Air Quality Classification

This project focuses on building and evaluating multiple **machine learning models** for classifying **air quality levels** based on environmental attributes.  
It helps in identifying whether the air is **Good, Moderate, Poor, or Hazardous**, which is crucial for environmental monitoring and public health awareness.

---

## 📌 Project Overview
- **Dataset**: Contains air quality measurements with multiple attributes.
- **Goal**: Predict the air quality class based on input features.
- **Classes**:  
  - Good  
  - Moderate  
  - Poor  
  - Hazardous  

---

## 🛠️ Technologies Used
- **Python**  
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn` (Logistic Regression, Decision Trees, Random Forest, SVM, KNN, Naive Bayes, PCA, GridSearchCV)
  - `imbalanced-learn (SMOTE)`
  - `PyTorch` (Neural Networks)
  - `scipy`

---

## 🔬 Methodology
1. **Data Preprocessing**  
   - Handle missing values  
   - Check class distribution  
   - Feature scaling (StandardScaler)  
   - Oversampling (SMOTE for imbalanced data)  

2. **Dimensionality Reduction**  
   - PCA applied for visualization and noise reduction.  

3. **Model Training & Evaluation**  
   Models used:
   - Logistic Regression  
   - Decision Tree Classifier  
   - Random Forest Classifier  
   - Support Vector Classifier (SVC)  
   - K-Nearest Neighbors (KNN)  
   - Naive Bayes  
   - Multi-Layer Perceptron (MLP)  
   - Neural Networks (PyTorch)  

4. **Evaluation Metrics**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  
   - Confusion Matrix  

---

## 📊 Results
- Compared multiple models using evaluation metrics.  
- Best performance achieved using **Random Forest and Neural Networks**.  
- PCA visualizations helped in understanding feature separability.  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/SOI_lunar-_vision.git
   cd SOI_lunar-_vision
