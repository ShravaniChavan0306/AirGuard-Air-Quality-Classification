# 🌿 AirGuard – AI-Powered Air Quality Classification System

**AirGuard** is a Machine Learning-based Air Quality Classification System developed using Python and Streamlit. It analyses air pollutant concentrations and predicts air quality categories using three different Machine Learning algorithms.

The project uses historical air quality data from India to train, evaluate, and compare different classification models. It also provides an interactive web dashboard where users can enter pollutant values, generate predictions, and compare model performance.

## 🚀 Features

- **Air Quality Prediction:** Predicts air quality based on pollutant concentration values.
- **Multiple Machine Learning Algorithms:** Implements three classification algorithms:
  - Random Forest
  - Support Vector Machine (SVM)
  - Neural Network (MLP)
- **Interactive Dashboard:** Provides a user-friendly Streamlit interface.
- **Model Comparison:** Compares models using Accuracy, Precision, Recall, and F1 Score.
- **Confusion Matrix Visualization:** Displays confusion matrices for all three models.
- **Prediction Probability:** Visualizes the probability of each predicted air quality category.
- **Best Model Identification:** Automatically identifies the best-performing model based on accuracy.
- **Data Visualization:** Uses interactive Plotly charts to display model performance.

## 🧠 Machine Learning Algorithms

### 1. Random Forest
Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification performance and prediction accuracy.

### 2. Support Vector Machine (SVM)
SVM is a supervised Machine Learning algorithm that identifies optimal decision boundaries to classify data into different categories. This project uses the RBF kernel.

### 3. Neural Network (MLP)
The Multilayer Perceptron (MLP) is a neural network algorithm that learns complex patterns through interconnected layers of computational units.

## 📊 Air Quality Categories

The system classifies air quality into three categories:

| Category | Description |
|---|---|
| 🌿 Good | Good air quality |
| 🌤️ Moderate | Moderate air quality |
| ⚠️ Poor | Poor air quality |

The original AQI categories are mapped into these three groups:
- Good and Satisfactory → Good
- Moderate → Moderate
- Poor, Very Poor, and Severe → Poor

## 🌫️ Input Features

The system uses the following nine air pollutant parameters:

| No. | Feature | Description |
|---|---|---|
| 1 | PM2.5 | Fine Particulate Matter |
| 2 | PM10 | Coarse Particulate Matter |
| 3 | NO | Nitric Oxide |
| 4 | NO2 | Nitrogen Dioxide |
| 5 | NOx | Nitrogen Oxides |
| 6 | NH3 | Ammonia |
| 7 | CO | Carbon Monoxide |
| 8 | SO2 | Sulfur Dioxide |
| 9 | O3 | Ozone |

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Saving:** Joblib
- **Web Framework:** Streamlit
- **Data Visualization:** Plotly, Matplotlib, Seaborn

## 📂 Project Structure

```text
AirGuard/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── data/
│   └── city_day.csv
│
├── models/
│   ├── random_forest.joblib
│   ├── svm.joblib
│   ├── neural_network.joblib
│   └── metrics.json
│
└── README.md
```

## 📥 Dataset

This project uses the **India Air Quality Dataset** available on Kaggle.

Dataset Link: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Download the dataset and place the `city_day.csv` file inside the `data` folder.

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AirGuard.git
```

### Step 2: Navigate to the Project Directory

```bash
cd AirGuard
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**For macOS/Linux:**

```bash
source venv/bin/activate
```

**For Windows:**

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Add the Dataset

Create a folder named `data` and place the downloaded `city_day.csv` file inside it.

### Step 7: Train the Machine Learning Models

Run the following command:

```bash
python train_model.py
```

This script will:
- Load and preprocess the dataset.
- Handle missing pollutant values using median imputation.
- Split the dataset into training and testing sets (80:20).
- Train all three Machine Learning algorithms.
- Evaluate model performance.
- Save trained models inside the `models` folder.
- Generate `metrics.json` containing evaluation results and default pollutant values.

### Step 8: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📈 Model Evaluation

The three Machine Learning models are evaluated using the following performance metrics:

| Metric | Description |
|---|---|
| Accuracy | Measures the overall correctness of predictions. |
| Precision | Measures the correctness of positive predictions. |
| Recall | Measures how effectively the model identifies actual classes. |
| F1 Score | Represents the harmonic mean of Precision and Recall. |
| Confusion Matrix | Shows correct and incorrect classifications. |

All evaluation results are displayed through the Model Comparison section of the dashboard.

## 🖥️ Application Modules

The AirGuard application contains four main sections:

1. **Dashboard:** Displays project information, the best-performing model, its accuracy, and model accuracy comparison.
2. **Predict Air Quality:** Allows users to enter pollutant concentrations, select an algorithm, and obtain an air quality prediction.
3. **Model Comparison:** Displays performance metrics, comparison charts, and confusion matrices.
4. **About Project:** Provides information about the project objectives, algorithms, dataset, input features, and technologies.

## 🎯 Project Objectives

- To analyse historical air pollution data from India.
- To apply different Machine Learning classification algorithms.
- To predict air quality based on pollutant concentrations.
- To compare the performance of multiple classification models.
- To develop an interactive web-based air quality prediction system.

## 🔮 Future Enhancements

- Integration of real-time air quality monitoring APIs.
- Location-based air quality prediction.
- Interactive maps for visualizing air pollution levels.
- Implementation of additional Machine Learning algorithms.
- Deployment of the application on a cloud platform.

## ⚠️ Disclaimer

AirGuard is an academic Machine Learning project developed for educational and demonstration purposes. It uses historical air quality data and does not provide live official AQI measurements, official CPCB classifications, or medical recommendations.

## 👨‍💻 Author

**Developed as a Machine Learning Academic Project.**

---

⭐ If you find this project interesting, consider giving the repository a star!
