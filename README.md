# Iris Flower Classification using Machine Learning and Deep Learning

This project implements and compares multiple **Machine Learning** and **Deep Learning** models on the classic **Iris Flower Dataset**.  
The objective is to analyze model performance, visualize decision boundaries, and study feature importance.

---

## 📌 Dataset
- **Name:** Iris Flower Dataset
- **Source:** Scikit-learn built-in dataset
- **Samples:** 150
- **Features:**
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Classes:**
  - Setosa
  - Versicolor
  - Virginica

---

## 🧠 Models Implemented

### Machine Learning Models
- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM – Linear Kernel)
- Naive Bayes (GaussianNB)
- K-Nearest Neighbors (KNN)

### Deep Learning Model
- Deep Neural Network (DNN / MLPClassifier)

---

## ⚙️ Methodology
1. Load and explore the dataset
2. Perform train–test split
3. Apply feature scaling where required
4. Train multiple ML and DL models
5. Evaluate models using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
6. Visualize:
   - Feature scatter plots
   - Decision boundaries (SVM)
   - Feature importance (DT, RF)
   - Model performance comparison

---

## 🧪 Deep Neural Network Details
- Architecture:
  - Input layer: 4 neurons
  - Hidden layers: Fully connected layers with ReLU activation
  - Output layer: 3 neurons with softmax
- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy
- Purpose:
  - To compare traditional ML models with a neural network approach
  - Demonstrate that simple datasets may not require deep architectures

---

## 📊 Results
- Linear SVM achieved the highest accuracy on the test set.
- Random Forest, and Naive Bayes showed stable and robust performance.
- Deep Neural Network achieved competitive accuracy but did not outperform simpler models.
- Feature importance analysis confirmed **petal length and petal width** as the most significant features.

---

## 📈 Visualizations Included
- Petal length vs petal width scatter plot
- Decision boundary plots:
  - Linear SVM
- Confusion matrices for all models
- Feature importance plots (Decision Tree & Random Forest)
- Bar chart comparing Accuracy, Precision, Recall, and F1-score

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/VidushiSharma31/iris-classification.git
cd iris-classification
```
### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the project

Open the Jupyter Notebook/Google colab

or

Run the Python script:

python iris_classification.ipynb

## 🛠️ Tools & Libraries

### Programming Language
- Python

### Data Handling & Visualization
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- Mlxtend

### Deep Learning
- TensorFlow (Keras API)

## 📌 Key Learnings

- Linear models can outperform complex models on linearly separable data
- Feature selection plays a critical role in visualization and performance
- Decision boundaries help explain model behavior
- Tree-based feature importance provides interpretability
- Deep neural networks are powerful but not always necessary for simple datasets
