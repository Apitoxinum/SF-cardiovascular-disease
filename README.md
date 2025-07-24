# SF-cardiovascular-disease

## Introduction
Welcome to the SF-cardiovascular-disease project repository. This project leverages machine learning to predict cardiovascular disease risk based on clinical parameters. With heart disease being a leading global cause of mortality, this system aims to assist in early detection and preventive healthcare.

## Objective
**Primary Goal:** Develop and compare machine learning models that can accurately identify individuals at high risk of heart disease using 13 key clinical features.

## Project Overview
This comprehensive analysis includes:
1. Data preprocessing and outlier handling
2. Exploratory Data Analysis (EDA) with visualizations
3. Correlation analysis of clinical features
4. Model training and evaluation of:
   - Logistic Regression
   - Random Forest Classifier
   - Neural Network (PyTorch)
5. Model deployment with inference script

## Dataset Features
[(https://www.kaggle.com/competitions/tech-weekend-data-science-hackathon/data)]

The dataset contains **600,000 training** and **400,000 test records** with these clinical parameters:

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Patient age | Numerical |
| `sex` | Gender (0=Female, 1=Male) | Categorical |
| `chest` | Chest pain type (1-4) | Categorical |
| `resting_blood_pressure` | Blood pressure at rest (mmHg) | Numerical |
| `serum_cholestoral` | Cholesterol level (mg/dl) | Numerical |
| `fasting_blood_sugar` | Fasting glucose >120 mg/dl | Binary |
| `resting_electrocardiographic_results` | ECG results at rest | Categorical |
| `maximum_heart_rate_achieved` | Max heart rate during test | Numerical |
| `exercise_induced_angina` | Exercise-triggered chest pain | Binary |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | ST segment slope during peak exercise | Categorical |
| `number_of_major_vessels` | Major vessels (0-3) in fluoroscopy | Categorical |
| `thal` | Thalassemia type (3,6,7) | Categorical |


###  Model Development
| Model                      | Framework      | 
|----------------------------|----------------|
| Logistic Regression        | Scikit-learn   |
| Random Forest Classifier   | Scikit-learn   |
| Neural Network (3-layer FC)| PyTorch        |

**Metrics tracked:**  
- Accuracy - Overall correctness
- F1 Score - Balance of precision/recall
- ROC-AUC - Class separation quality


## 🔍 Статья о применении моделей машинного обучения для предсказания ССЗ:
[(https://www.sciencedirect.com/science/article/pii/S2772442522000016)}

## English Version

### Title
**An Artificial Intelligence Model for Heart Disease Detection Using Machine Learning Algorithms**

### Authors
Victor Chang, Vallabhaneni Rupa Bhavani, Ariel Qianwen Xu, M.A. Hossain

### Description
This research presents a Python-based AI system for early detection of heart disease, leveraging machine learning (ML). The core methodology employs a **Random Forest Classifier**, trained and evaluated on a clinical dataset containing 14 key patient attributes (e.g., age, cholesterol, chest pain type). 

The study details critical steps including:
- Data preprocessing (handling categorical variables)
- Model development following software engineering practices
- Rigorous performance evaluation

The system achieved an initial **accuracy of approximately 83%**, demonstrating significant potential for aiding medical diagnosis. Key contributions include the practical implementation framework and the validation of ML, particularly Random Forest, for improving predictive accuracy in cardiovascular healthcare.

**Published under Creative Commons license (Open Access).**

### Key Highlights
- **Core Tech**: AI/ML (Random Forest Classifier) for heart disease prediction
- **Implementation**: Python-based application using Pandas, Scikit-learn, etc.
- **Focus**: Data preprocessing, model development lifecycle, performance validation
- **Result**: ~83% accuracy on training data, with potential for improvement
- **Impact**: Aids early detection, supports clinical decision-making

---

## 🇷🇺 Русская Версия

### Название
**Модель искусственного интеллекта для выявления сердечных заболеваний с использованием алгоритмов машинного обучения**

### Авторы
Виктор Чанг, Валлабханени Рупа Бхавани, Ариэль Цяньвэнь Сюй, М.А. Хоссайн

### Описание
В этой статье представлена **система на основе искусственного интеллекта (ИИ)**, разработанная на Python для **раннего выявления сердечных заболеваний**. В основе системы лежат алгоритмы **машинного обучения**, главным образом — **метод Random Forest (Случайный лес)**.

Исследование подробно описывает ключевые этапы:
- Подготовку медицинских данных (включая работу с категориальными признаками пациентов)
- Разработку и обучение модели
- Оценку ее эффективности

Разработанная модель продемонстрировала **точность предсказания около 83%**. Работа подчеркивает практическую ценность машинного обучения и языка Python для создания инструментов, способных **помочь врачам в диагностике** сердечно-сосудистых заболеваний на основе анализа клинических показателей.

**Статья опубликована в открытом доступе (Creative Commons).**

### Ключевые аспекты
- **Технология**: Искусственный интеллект и машинное обучение (Random Forest)
- **Реализация**: Практическая система на Python (Pandas, Scikit-learn и др.)
- **Задача**: Автоматическое выявление риска сердечных заболеваний по данным пациентов
- **Результат**: Точность системы ~83%, что подтверждает ее эффективность
- **Польза**: Создание инструмента для поддержки врачебной диагностики и раннего выявления
