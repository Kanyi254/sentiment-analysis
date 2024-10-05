# sentiment-analysis
# Twitter Sentiment Analysis for Apple and Google Products

This project analyzes Twitter sentiment towards Apple and Google products using machine learning.

## Project Overview

This project uses a CrowdFlower dataset of tweets about Apple and Google products. It performs sentiment analysis to classify tweets as positive, negative, or neutral.

The project involves:

* Data cleaning and preprocessing
* Exploratory data analysis 
* Feature extraction using TF-IDF
* Model training using various algorithms (Logistic Regression, Random Forest, SVM, etc.)
* Model evaluation and selection
* Handling class imbalance with SMOTE

## Key Findings

* The best performing models were Logistic Regression and Random Forest, both achieving an accuracy of 83.7%.
* TF-IDF vectorization proved superior to CountVectorizer for feature extraction.

## Recommendations

* Focus on converting neutral sentiment to positive.
* Address negative feedback promptly.
* Engage neutral customers to potentially shift sentiment.

## Files

* `judge-1377884607_tweet_product_company.csv`: The dataset used for analysis.
* `notebook.ipynb`: Jupyter notebook containing the code and analysis.
* `models/`: Directory containing trained models and other resources.

## How to Run

1.  Open the `notebook.ipynb` file in Google Colab.
2.  Run all the cells in the notebook.

## Dependencies

* Python 3.x
* Pandas
* Scikit-learn
* NLTK
* Imbalanced-learn
* WordCloud
