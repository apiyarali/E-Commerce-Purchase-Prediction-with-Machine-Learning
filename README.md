# Online Shopping Purchase Prediction using KNN

## Overview
This project implements an AI model to predict whether an online shopping customer will complete a purchase. The model uses a k-nearest neighbors (KNN) classifier to analyze user behavior based on various browsing metrics.

## Features
- Loads and processes online shopping session data from `shopping.csv`
- Trains a k-nearest neighbors classifier (K=1) to predict purchase intent
- Evaluates the model's accuracy using sensitivity (true positive rate) and specificity (true negative rate)
- Outputs performance metrics to the console

## Usage
### Prerequisites
Ensure you have Python installed along with the required dependencies:
```sh
pip install scikit-learn
```

### Running the Program
To execute the program, run the following command:
```sh
python shopping.py shopping.csv
```
### Expected Output
```
Correct: 4088
Incorrect: 844
True Positive Rate: 41.02%
True Negative Rate: 90.55%
```

## Data Format
The dataset (`shopping.csv`) consists of approximately 12,000 user sessions, each containing 17 evidence columns and a `Revenue` column (1 for purchase, 0 for no purchase). The features include:
- **Page visits & duration:** Number of pages visited and time spent (Administrative, Informational, Product-related)
- **Google Analytics metrics:** Bounce rates, exit rates, and page values
- **Shopping context:** SpecialDay indicator, month of visit, weekend status
- **User attributes:** Operating system, browser, region, traffic type, visitor type

## Implementation Details
### `load_data(filename)`
- Reads the CSV file and extracts evidence and labels
- Converts categorical data to numeric format (e.g., months to integers, boolean values to 0/1)

### `train_model(evidence, labels)`
- Trains a `KNeighborsClassifier` with `K=1` using the given dataset

### `evaluate(labels, predictions)`
- Computes sensitivity (true positive rate) and specificity (true negative rate) based on classifier predictions

## Model Performance
- **Sensitivity (True Positive Rate):** 41.02%
- **Specificity (True Negative Rate):** 90.55%
- The classifier successfully identifies non-purchasing users with high accuracy but has room for improvement in detecting purchasing users.

## References
- Dataset: [Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018)](https://link.springer.com/article/10.1007/s00521-018-3523-0)
- Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)

## License
This project is part of an Harvard's CS80 AI coursework assignment exploring Machine Learning.

---
### Future Improvements
- Experiment with different values of K in KNN
- Use additional machine learning models (e.g., decision trees, logistic regression)
- Feature engineering to improve sensitivity for purchase detection

