# 📊 Early Prediction of Year 3 Reading Risk Using Machine Learning

## Overview

This project develops a machine learning–based early-intervention screening approach to identify primary school students who may be at risk of underperforming in the **Year 3 NAPLAN Reading assessment**.

The analysis uses student information from **Year 1 and Year 2**, including literacy and numeracy assessments, demographic characteristics, family background, disability information, and school socioeconomic indicators.

The project combines:

* Exploratory Data Analysis (EDA)
* Data quality assessment and preprocessing
* Supervised machine learning classification
* Logistic Regression
* Random Forest
* Model comparison and evaluation
* K-Means clustering
* Principal Component Analysis (PCA)
* Student risk-profile segmentation

The overall objective is to shift educational support from a **reactive approach after Year 3 results** to a more **proactive early-intervention approach** using Year 1–2 information.

---

## 🎯 Business Problem

Schools need to identify students who may require additional literacy support before formal Year 3 NAPLAN outcomes are available.

The key business question addressed in this project is:

> **Can Year 1–2 academic, demographic, family, and school-SES information be used to identify students who are at risk of underperforming in Year 3 Reading?**

An effective screening model could help schools:

* Identify potentially at-risk students earlier
* Support targeted literacy interventions
* Prioritise limited educational resources
* Understand different student risk profiles
* Move from reactive to proactive intervention

---

## 📁 Dataset

The dataset contains **2,000 student records and 34 variables**.

The features represent several categories of information:

| Category             | Examples                                              |
| -------------------- | ----------------------------------------------------- |
| Literacy             | TextLevel, WritingVocab, HRSIW                        |
| Numeracy             | Counting, Place Value, Addition & Subtraction         |
| Demographics         | Gender, Kindergarten Age                              |
| Disability & Support | Disability categories, NCCD-Funded                    |
| Family Background    | Number of siblings, sibling order, parental education |
| School Context       | School SES indicators                                 |
| Target               | Year3_Reading_At_Risk                                 |

### Target Variable

`Year3_Reading_At_Risk`

The target contains two classes:

* `False` — student not classified as at risk
* `True` — student classified as at risk

The dataset contains:

* **1,600 non-at-risk students (80%)**
* **400 at-risk students (20%)**

Because the target is imbalanced, model evaluation places particular importance on **recall, F1-score and ROC-AUC**, rather than accuracy alone.

---

## 🔍 Project Workflow

```text
Data Loading
     ↓
Data Quality Assessment
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Preparation
     ↓
Stratified Train/Test Split
     ↓
Preprocessing
     ↓
Logistic Regression ───┐
                       ├── Model Comparison
Random Forest ─────────┘
     ↓
Risk Prediction
     ↓
K-Means Clustering
     ↓
PCA Visualisation
     ↓
Student Risk-Profile Segmentation
```

---

## 🧹 Data Quality & Preprocessing

The dataset was assessed for:

* Missing values
* Duplicate records
* Duplicate student IDs
* Out-of-range values
* Logical inconsistencies

### Data Quality Findings

No missing values or duplicate student records were identified.

Several out-of-range values were found in literacy and family-background variables. These were handled using domain-based clipping rather than deleting observations.

Examples include:

* `TextLevel` values clipped to the valid **0–31** range
* Negative `HRSIW-01-SOY` values clipped to zero
* `NumAbvYear9`, `NumAbvDiploma` and `NumProf` values capped at two where required

This approach preserved the student records while correcting values that were outside their expected ranges.

---

## 📈 Exploratory Data Analysis

The analysis investigated relationships between student characteristics and Year 3 reading risk.

Major areas explored included:

### Literacy Development

Year 1 and Year 2 reading assessment scores were compared between at-risk and non-at-risk students.

The analysis also examined the student's reading development trajectory between Year 1 and Year 2.

### Numeracy

Numeracy indicators including:

* Counting
* Place Value
* Addition and Subtraction
* Multiplication and Division

were examined against reading-risk status.

### Demographic Factors

The project investigated:

* Gender
* Kindergarten age
* Disability categories
* NCCD funding

### Socioeconomic Factors

School SES indicators were analysed to examine differences between students classified as at risk and those not classified as at risk.

### Correlation Analysis

A correlation heatmap and ranked feature correlations were used to investigate relationships between numerical predictors and the target variable.

---

# 🤖 Machine Learning Models

Two supervised classification models were developed.

## 1. Logistic Regression

Logistic Regression was implemented as an interpretable baseline classification model.

The model was trained using:

* Standardised numerical variables
* One-hot encoded categorical variables
* `class_weight='balanced'`
* 5-fold cross-validation
* Grid Search for hyperparameter tuning

### Best Hyperparameter

```text
C = 0.01
```

### Test Performance

| Metric              | Result |
| ------------------- | -----: |
| Accuracy            |   0.78 |
| Precision — At Risk |   0.47 |
| Recall — At Risk    |   0.79 |
| F1-score — At Risk  |   0.59 |
| ROC-AUC             |  0.867 |

The model achieved a **0.867 ROC-AUC** and identified approximately **79% of the students classified as at risk** in the test set.

---

## 2. Random Forest

A Random Forest classifier was developed to capture potentially nonlinear relationships and interactions between student characteristics.

Hyperparameters were tuned using Grid Search and 5-fold cross-validation.

### Best Hyperparameters

```text
n_estimators = 400
max_depth = 5
min_samples_leaf = 5
```

### Test Performance

| Metric              | Result |
| ------------------- | -----: |
| Accuracy            |   0.79 |
| Precision — At Risk |   0.49 |
| Recall — At Risk    |   0.71 |
| F1-score — At Risk  |   0.58 |
| ROC-AUC             |  0.854 |

---

## ⚖️ Model Comparison

| Metric            | Logistic Regression | Random Forest |
| ----------------- | ------------------: | ------------: |
| Accuracy          |                0.78 |          0.79 |
| At-Risk Precision |                0.47 |          0.49 |
| At-Risk Recall    |            **0.79** |          0.71 |
| At-Risk F1        |                0.59 |          0.58 |
| ROC-AUC           |           **0.867** |         0.854 |

For this project's early-intervention context, recall is particularly important because failing to identify an at-risk student could result in missed support opportunities.

Logistic Regression therefore provides the stronger combination of recall and ROC-AUC in this analysis, while Random Forest provides slightly higher accuracy and precision.

---

# 👥 Student Risk Segmentation

In addition to prediction, **K-Means clustering** was used to identify groups of students with similar characteristics.

Clustering was evaluated using:

* Within-Cluster Sum of Squares (WCSS)
* Silhouette Score
* Davies-Bouldin Index
* Risk-rate separation across clusters

Although clustering metrics were considered alongside business relevance, **K = 4** was selected for the final student segmentation.

### Final Cluster Profiles

| Cluster   | Students | Risk Rate | Profile            |
| --------- | -------: | --------: | ------------------ |
| Cluster 0 |      630 |      8.4% | Moderate-Low Risk  |
| Cluster 1 |      422 |      1.2% | Lowest Risk        |
| Cluster 2 |      541 |     25.5% | Moderate-High Risk |
| Cluster 3 |      407 |     50.1% | Highest Risk       |

### Key Segmentation Insight

**Cluster 3** has the highest observed risk rate at **50.1%**.

This cluster also shows:

* Lowest average Year 1 TextLevel
* Lowest average Year 2 TextLevel
* Lowest average Counting score
* Highest proportion of NCCD-funded students

This segmentation can help demonstrate how students with different characteristics may require different levels or types of intervention.

---

# 📊 PCA Visualisation

Principal Component Analysis (PCA) was used to reduce the processed feature space to two dimensions for visualisation.

The PCA projection provides a visual representation of how the four student clusters are distributed across the main dimensions of variation in the dataset.

---

# 🛠️ Technologies & Libraries

The project was developed in Python using:

* **Python**
* **Pandas** — data manipulation and analysis
* **NumPy** — numerical computation
* **Matplotlib** — visualisation
* **Seaborn** — statistical visualisation
* **Scikit-learn** — preprocessing, modelling and evaluation
* **Google Colab** — development environment

### Main Machine Learning Techniques

```text
Data Preprocessing
├── StandardScaler
├── OneHotEncoder
└── ColumnTransformer

Supervised Learning
├── Logistic Regression
└── Random Forest

Model Selection
└── GridSearchCV

Evaluation
├── Precision
├── Recall
├── F1-score
├── ROC-AUC
└── Confusion Matrix

Unsupervised Learning
└── K-Means

Dimensionality Reduction
└── PCA
```

---

# 📌 Key Findings

1. **Year 1–2 student information can provide useful signals for identifying Year 3 reading risk.**

2. The dataset contains a **20% at-risk class**, making class imbalance an important modelling consideration.

3. Logistic Regression achieved a **ROC-AUC of 0.867** on the test set.

4. Logistic Regression achieved **0.79 recall for the at-risk class**, which is important for an early-intervention screening application.

5. Random Forest achieved slightly higher accuracy (**0.79 vs 0.78**) but lower recall and ROC-AUC than Logistic Regression.

6. K-Means segmentation identified four student profiles with substantially different observed risk rates.

7. The highest-risk cluster had a **50.1% observed risk rate**, compared with only **1.2% in the lowest-risk cluster**.

8. The combination of prediction and segmentation provides two complementary perspectives: **who may be at risk** and **what types of student profiles exist within the population**.

---

# 💡 Business Value

The project demonstrates how educational data can potentially support earlier and more targeted intervention.

Instead of waiting for Year 3 NAPLAN results, schools could use earlier academic and contextual information to identify students who may benefit from additional support.

A potential decision-support workflow could be:

```text
Student Data
     ↓
Risk Probability
     ↓
Early Identification
     ↓
Teacher Review
     ↓
Targeted Literacy Support
     ↓
Progress Monitoring
```

Importantly, the model should be considered a **screening and decision-support tool**, rather than an automatic decision-making system. Teacher judgement and contextual information remain important when determining appropriate student support.

---

# ⚠️ Limitations

Several limitations should be considered:

* The dataset contains 2,000 observations and may not represent every Australian school population.
* Model performance is based on a single train/test split.
* Predictive relationships do not establish causal relationships.
* A predicted risk score should not be interpreted as a definitive outcome for an individual student.
* Clustering results depend on preprocessing choices and the selected number of clusters.
* Further external validation would be required before using the model in a real-world educational environment.

---

# 📂 Repository Structure

A recommended repository structure is:

```text
MIS710-Machine-Learning-Reading-Risk/
│
├── README.md
│
├── MIS710A2_Python_Ravalji_Dhruvrajsinh_226491165.ipynb
│
├── data/
│   └── README.md
│
├── images/
│   ├── ml_process_diagram.png
│   ├── model_comparison.png
│   └── cluster_visualisation.png
│
└── requirements.txt
```

> The original dataset is not included in this repository unless permission is available to redistribute it.

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MIS710-Machine-Learning-Reading-Risk.git
```

### 2. Open the notebook

Open:

```text
MIS710A2_Python_Ravalji_Dhruvrajsinh_226491165.ipynb
```

using **Jupyter Notebook**, **JupyterLab**, or **Google Colab**.

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Update the dataset path

The original notebook loads the dataset from Google Drive. If running outside the original environment, update the CSV path to the location of the dataset on your machine.

### 5. Run the notebook

Execute the notebook cells sequentially to reproduce:

* Data quality checks
* EDA
* Data preprocessing
* Logistic Regression
* Random Forest
* Model evaluation
* K-Means clustering
* PCA visualisation

---

# 🧠 Skills Demonstrated

This project demonstrates practical skills in:

**Data Analytics**

* Data cleaning
* Data quality assessment
* Exploratory data analysis
* Correlation analysis
* Data visualisation

**Machine Learning**

* Classification
* Logistic Regression
* Random Forest
* K-Means clustering
* PCA
* Hyperparameter tuning
* Cross-validation

**Model Evaluation**

* Confusion matrices
* Precision
* Recall
* F1-score
* ROC-AUC
* Silhouette Score
* Davies-Bouldin Index

**Business Analytics**

* Translating a business problem into an ML problem
* Early-intervention decision support
* Risk segmentation
* Interpreting model results for stakeholders

---

# 🤖 Generative AI Use

Generative AI was used as a supplementary learning and development aid during the project.

Its use included:

* Understanding dataset variables
* Clarifying Python and machine learning code
* Understanding analytical terminology
* Supporting comprehension of model evaluation techniques
* Improving understanding of the logical flow of the analysis

The analytical decisions, model selection, implementation and interpretation of project results were undertaken as part of the student's work.

---



## ⭐ Project Summary

> **An end-to-end machine learning project using Year 1–2 student data to predict Year 3 reading risk and segment students into actionable risk profiles using supervised and unsupervised learning.**
