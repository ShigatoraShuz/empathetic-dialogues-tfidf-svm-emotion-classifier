# Empathetic Dialogues TF-IDF + Linear SVM Emotion Classifier

A Google Colab-ready machine learning pipeline for **text-based emotion classification** using the **EmpatheticDialogues dataset** mapped into **Plutchik-inspired emotion categories**. This project uses **TF-IDF feature extraction** and a **Linear Support Vector Machine (Linear SVM)** classifier to identify emotions from text.

The pipeline includes dataset downloading, preprocessing, label mapping, group-safe train-validation-test splitting, model training, hyperparameter comparison, evaluation metrics, classification reports, and confusion matrix visualizations.

---

## Project Overview

This project aims to build a lightweight and research-ready emotion classification model that can classify user text into eight emotion categories based on Plutchik’s emotion model.

The model uses a classical machine learning approach instead of transformer-based deep learning models. It is designed to be simple, fast, interpretable, and suitable for Google Colab execution.

---

## Emotion Categories

The dataset labels are mapped into the following Plutchik-inspired emotion categories:

| Class No. | Emotion Category |
|----------|------------------|
| 0 | Anger |
| 1 | Anticipation |
| 2 | Disgust |
| 3 | Fear |
| 4 | Joy |
| 5 | Sadness |
| 6 | Surprise |
| 7 | Trust |

---

## Features

- Google Colab-ready training pipeline
- Kaggle dataset download support
- Uses EmpatheticDialogues dataset
- Maps original emotion labels into Plutchik emotion categories
- Cleans and prepares text data
- Uses group-safe train, validation, and test splitting
- Uses TF-IDF feature extraction
- Trains multiple Linear SVM configurations
- Selects best model based on validation macro F1-score
- Evaluates performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Macro average
  - Weighted average
  - Support per class
- Generates visualizations:
  - Label distribution chart
  - Validation comparison chart
  - Overall test metrics chart
  - Per-category F1-score chart
  - Raw confusion matrix
  - Normalized confusion matrix
- Saves trained model artifacts using `joblib`
- Includes a sample prediction function

---

## Dataset

This project uses the **EmpatheticDialogues** dataset from Kaggle.

The dataset contains conversational text samples labeled with emotion-related categories. These labels are processed and mapped into eight Plutchik-inspired emotion classes.

### Expected Dataset File

The pipeline automatically searches for CSV files after downloading the dataset. The expected selected file is usually:

```text
emotion-emotion_69k.csv
```

### Important Columns Used

The pipeline checks and processes columns such as:

```text
Situation
emotion
empathetic_dialogues
labels
Unnamed columns
```

The final processed dataset keeps only the cleaned text and the mapped Plutchik emotion label.

---

## Methodology

The training process follows this workflow:

```text
1. Install required libraries
2. Import dependencies
3. Set global configuration
4. Download dataset from Kaggle
5. Load dataset
6. Map original labels to Plutchik emotion categories
7. Clean dataset
8. Extract customer text
9. Create leakage-safe situation groups
10. Remove conflicting groups
11. Prepare final text input
12. Visualize label distribution
13. Perform group-safe train-validation-test split
14. Encode labels
15. Prepare text arrays
16. Apply TF-IDF feature extraction
17. Train Linear SVM configurations
18. Evaluate on validation set
19. Select best model using validation macro F1-score
20. Evaluate final model on test set
21. Generate classification report
22. Compute per-category metrics
23. Save model artifacts
24. Test sample prediction function
```

---

## Model Architecture

This project uses a traditional machine learning architecture:

```text
Raw Text
   ↓
Text Cleaning
   ↓
Emotion Label Mapping
   ↓
TF-IDF Vectorization
   ↓
Linear SVM Classifier
   ↓
Emotion Prediction
```

### Feature Extraction

The project uses **TF-IDF Vectorizer** to convert text into numerical features.

Two types of TF-IDF features are used:

1. Word-level TF-IDF
2. Character-level TF-IDF

These feature vectors are combined before being passed into the Linear SVM classifier.

---

## Machine Learning Algorithm

The main classifier used in this project is:

```text
Linear Support Vector Machine
```

Implemented using:

```python
sklearn.svm.LinearSVC
```

Linear SVM is suitable for high-dimensional sparse text data, especially when combined with TF-IDF features.

---

## Training Configurations

The pipeline trains and compares multiple Linear SVM configurations using different values of:

```text
C
class_weight
```

Example configurations:

```python
[
    {"C": 0.3, "class_weight": "balanced"},
    {"C": 0.5, "class_weight": "balanced"},
    {"C": 0.3, "class_weight": None}
]
```

The best model is selected based on:

```text
Validation Macro F1-score
```

Macro F1-score is used because the dataset contains class imbalance, and macro F1 gives equal importance to each emotion category.

---

## Evaluation Metrics

The model is evaluated using the following metrics:

| Metric | Description |
|-------|-------------|
| Accuracy | Overall percentage of correct predictions |
| Precision | How many predicted samples were correct |
| Recall | How many actual samples were correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Macro Average | Average score across all classes equally |
| Weighted Average | Average score weighted by class support |
| Support | Number of samples per class |

---

## Sample Test Results

The sample run from the provided pipeline produced approximately the following overall test performance:

| Metric | Score |
|-------|-------|
| Accuracy | 0.7000 |
| Macro Precision | 0.6872 |
| Macro Recall | 0.6716 |
| Macro F1-score | 0.6768 |
| Weighted Precision | 0.7103 |
| Weighted Recall | 0.7000 |
| Weighted F1-score | 0.7008 |

These results show that the TF-IDF + Linear SVM model achieved around **70% overall test accuracy** on the emotion classification task.

---

## Output Files

After training, the pipeline saves the trained model and related artifacts.

Example output files:

```text
linear_svm_plutchik_model.pkl
tfidf_word_vectorizer.pkl
tfidf_char_vectorizer.pkl
label_encoder.pkl
training_results.csv
classification_report.csv
per_category_results.csv
overall_test_metrics.csv
confusion_matrix_seaborn.png
normalized_confusion_matrix_seaborn.png
```

---

## Visualizations

The pipeline generates the following visual outputs:

### 1. Label Distribution

Shows the number of samples per Plutchik emotion category.

### 2. Top Linear SVM Validation Macro F1

Compares the validation macro F1-score of the top Linear SVM configurations.

### 3. Overall Test Metrics

Displays overall accuracy, precision, recall, F1-score, macro average, and weighted average.

### 4. Per-Category F1-score

Shows the F1-score of each emotion category.

### 5. Confusion Matrix

Displays the raw confusion matrix using Seaborn.

### 6. Normalized Confusion Matrix

Displays the normalized confusion matrix to better analyze class-level prediction behavior.

---

## Installation

This project is designed to run on Google Colab.

Install the required libraries:

```python
!pip install -q kagglehub scikit-learn pandas numpy matplotlib seaborn joblib
```

---

## How to Run in Google Colab

1. Open Google Colab.
2. Create a new notebook.
3. Copy and paste the full training pipeline.
4. Run all cells from top to bottom.
5. Wait for the dataset to download and extract.
6. Train the TF-IDF + Linear SVM model.
7. Review the evaluation results.
8. Download the saved model artifacts.

---

## Kaggle Dataset Download

The pipeline uses `kagglehub` to download the dataset.

Example:

```python
import kagglehub

path = kagglehub.dataset_download(
    "atharvjairath/empathetic-dialogues-facebook-ai"
)
```

After downloading, the pipeline automatically searches for CSV files and selects the dataset file.

---

## Requirements

The main Python libraries used are:

```text
kagglehub
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
scipy
```

Optional:

```text
Google Colab
Kaggle account
```

---

## Project Structure

Recommended GitHub repository structure:

```text
empathetic-dialogues-tfidf-svm-emotion-classifier/
│
├── README.md
├── notebook/
│   └── empathetic_dialogues_tfidf_svm_emotion_classifier.ipynb
│
├── outputs/
│   ├── training_results.csv
│   ├── classification_report.csv
│   ├── per_category_results.csv
│   ├── overall_test_metrics.csv
│   ├── confusion_matrix_seaborn.png
│   └── normalized_confusion_matrix_seaborn.png
│
├── models/
│   ├── linear_svm_plutchik_model.pkl
│   ├── tfidf_word_vectorizer.pkl
│   ├── tfidf_char_vectorizer.pkl
│   └── label_encoder.pkl
│
├── docs/
│   └── training_process_documentation.pdf
│
└── requirements.txt
```

---

## Example Prediction

The trained model can be used to predict the emotion of new text.

Example input:

```python
sample_text = "I feel alone, guilty, nervous, and disappointed with myself."
```

Example output:

```text
Predicted emotion: sadness
```

---

## Research Significance

Emotion classification is an important natural language processing task that can support applications such as:

- Mental health journaling systems
- Emotional well-being dashboards
- Sentiment monitoring tools
- Conversational agents
- Educational emotion analysis
- Human-computer interaction systems

This project demonstrates that a classical machine learning pipeline using TF-IDF and Linear SVM can still provide competitive baseline performance for emotion classification without requiring large transformer models.

---

## Strengths of the Approach

- Lightweight and fast to train
- Works well in Google Colab
- Does not require GPU
- Easier to interpret than deep learning models
- Good baseline for emotion classification
- Suitable for academic and research demonstrations
- Produces complete evaluation metrics and visualizations

---

## Limitations

Although the model achieved useful performance, it has several limitations:

1. The model depends heavily on the quality of label mapping.
2. TF-IDF does not deeply understand context like transformer-based models.
3. Some emotions may overlap semantically, causing misclassification.
4. Class imbalance can affect minority categories.
5. The model may not generalize well to text outside the dataset domain.
6. The dataset contains conversational examples, so performance may differ on journal-style text.

---

## Future Improvements

Possible improvements include:

- Testing additional classical models such as Logistic Regression, Naive Bayes, and Random Forest
- Using ensemble learning
- Applying oversampling or class balancing techniques
- Improving text preprocessing
- Expanding the dataset
- Comparing results with transformer-based models
- Testing on real journal entries
- Adding explainability tools
- Deploying the model as a web API
- Integrating the classifier into an emotion tracking dashboard

---

## Suggested Repository Name

```text
empathetic-dialogues-tfidf-svm-emotion-classifier
```

---

## Suggested Repository Description

```text
Google Colab-ready TF-IDF + Linear SVM emotion classification pipeline using EmpatheticDialogues mapped to Plutchik emotion labels, with full evaluation metrics and Seaborn confusion matrix visualization.
```

---

## How to Cite This Project

If this repository is used for academic work, cite it as:

```text
Punzalan, L. (2026). Empathetic Dialogues TF-IDF + Linear SVM Emotion Classifier. GitHub repository.
```

---

## Author

**Lloyd Punzalan**

---

## License

This project is intended for academic and research purposes.

Recommended license:

```text
MIT License
```

---

## Disclaimer

This project is for research and educational purposes only. It is not intended to provide clinical, psychological, or medical diagnosis. If used in a mental health-related application, the model output should be treated as supportive insight only and must not replace professional evaluation or intervention.
