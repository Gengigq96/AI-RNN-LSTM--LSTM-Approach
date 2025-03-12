# Handling Imbalanced Data with Recurrent Neural Networks (RNN) – LSTM Approach

This project explores how to train **Recurrent Neural Networks (RNNs) with LSTM** while addressing the challenge of **imbalanced data**. Instead of collecting additional data, we experiment with different strategies to handle class imbalance effectively.

## 📌 Project Overview

- **Deep Learning Model**: LSTM-based Recurrent Neural Network.
- **Goal**: Classify text into predefined categories.
- **Key Challenge**: The dataset is highly imbalanced, affecting classification accuracy.
- **Techniques Applied**:
  1. Weighted Loss Function
  2. Data Rebalancing (Oversampling & Undersampling)
  3. Architectural Improvements
  4. Reformulating the Problem
  5. Dimensionality Reduction of Classes

## 🛠 Technologies Used

- **Python** – Programming language.
- **TensorFlow & Keras** – Deep learning framework.
- **NumPy & Pandas** – Data manipulation.
- **Matplotlib & Seaborn** – Visualization.

## ⚙️ Strategies to Handle Imbalanced Data

### 1️⃣ Weighted Loss Function

By default, the model uses **`sparse_categorical_crossentropy`**, treating all classes equally. To improve classification of **minority classes**, we apply **Class Weights**:

- **Higher weight for underrepresented classes** to penalize misclassification.
- Implemented in TensorFlow using a **class weight dictionary**.
- Helps the model **focus more on minority classes**.

### 2️⃣ Data Rebalancing

**Imbalanced data** can lead to poor model performance. We apply two techniques:

#### ✅ Oversampling (Increasing Minority Class Samples)
- **Duplicate** examples from minority classes.
- **Text Data Augmentation**: Generate synonyms or rephrase sentences.
- Improves model’s ability to **learn from limited minority class data**.

#### ❌ Undersampling (Reducing Majority Class Samples)
- **Randomly remove samples** from overrepresented classes.
- Reduces bias toward dominant classes.

Both techniques can be **combined** for optimal balance.

### 3️⃣ Architectural Improvements

A **single-layer LSTM** may not effectively capture patterns in imbalanced data. We experiment with:

- **Additional LSTM Layers** for better contextual learning.
- **Extra Dense Layers** for improved classification.

### 4️⃣ Reformulating the Problem

Since our target variable is **ordinal categorical**, we explore an alternative **regression approach**:

- **Predict intensity as a continuous variable** instead of categorical classes.
- Helps capture **gradual variations** between intensity levels.

### 5️⃣ Dimensionality Reduction of Classes

To simplify the classification task, we:

- **Analyze patterns in class distributions**.
- **Merge similar classes** to reduce the number of target categories.
- **Improve model performance** by focusing on meaningful distinctions.