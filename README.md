# sentiment-analysis
* members
* Zackariah Komu
* Veona Maina
* Michael Omondi
* Maureen Imanene
# Twitter Sentiment Analysis for Apple and Google Products
[Slides](https://www.canva.com/design/DAGSuyFFjZg/f0hEF9yz84lV4yedymKk3A/edit?utm_content=DAGSuyFFjZg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
## Summary
This project aims to perform **sentiment analysis** on Twitter data for **Apple** and **Google** products. The goal is to classify tweets into positive, negative, or neutral sentiments to understand how customers perceive these products. The analysis involves **binary classification** and **multiclass classification** using various machine learning models such as **Logistic Regression**, **Random Forest**, **Support Vector Machines (SVM)**, and **Naive Bayes**. This study will help businesses gain insights into customer feedback and adjust their marketing and product strategies accordingly.

## Business Understanding
Understanding customer sentiment is crucial for companies like Apple and Google to improve product offerings and enhance customer satisfaction. Sentiment analysis provides value in the following areas:
- **Product Improvements**: Identifying positive and negative aspects of products to enhance customer satisfaction.
- **Marketing Strategies**: Leveraging positive feedback for marketing campaigns while addressing negative feedback to improve brand perception.
- **Brand Health Monitoring**: Tracking customer loyalty and brand reputation through social media.

## Data Understanding
The dataset used in this project is sourced from **CrowdFlower** and contains **Twitter sentiment data** related to Apple and Google products. Key features in the dataset include:
- `tweet_text`: The main text of the tweet (free text).
- `emotion_in_tweet_is_directed_at`: The brand or product mentioned in the tweet.
- `is_there_an_emotion_directed_at_a_brand_or_product`: Labeled sentiment categories such as **positive**, **negative**, **neutral**, and **can't tell**.

## Project Workflow
1. **Data Preprocessing**:
   - Text cleaning (removal of punctuation, lowercasing, etc.).
   - Handling missing values and duplicates.
   - Tokenization and stopword removal.
2. **Exploratory Data Analysis (EDA)**:
   - Visualization of sentiment distributions and word frequency analysis.
   - Word cloud generation for understanding common terms.
3. **Feature Engineering**:
   - Feature extraction using **TF-IDF Vectorization** and **CountVectorizer**.
   - Addition of custom features like **word count**, **char count**, **polarity**, and **subjectivity**.
4. **Modeling**:
   - Machine learning models implemented for both **binary** and **multiclass** classification:
     - **Logistic Regression**
     - **Random Forest**
     - **Support Vector Machines (SVM)**
     - **Naive Bayes**
     - **Decision Trees**

## Model Results

### **1. Binary Classification**
- **Logistic Regression**:
  - Accuracy: 0.90
  - Precision (Class 1): 0.95, Recall (Class 1): 0.94

- **Support Vector Machines (SVM)**:
  - Accuracy: 0.91
  - Precision (Class 1): 0.92, Recall (Class 1): 0.99

### **2. Multiclass Classification**
- **Multinomial Naive Bayes (Count Vectorizer)**:
  - Accuracy: 0.603, Recall: 0.604
  - F1-Score (Class 0): 0.63, F1-Score (Class 2): 0.57

- **Multinomial Naive Bayes (TF-IDF)**:
  - Accuracy: 0.758, Recall: 0.757
  - F1-Score (Class 0): 0.90, F1-Score (Class 2): 0.71

### **3. Model Comparison**:
| Model                    | Accuracy | Recall |
|---------------------------|----------|--------|
| RandomForest              | 0.710    | 0.710  |
| Tuned Decision Tree        | 0.754    | 0.753  |
| Multinomial Naive Bayes    | 0.765    | 0.764  |
| Tuned Multinomial Naive Bayes | 0.800 | 0.800  |
| Logistic Regression        | 0.808    | 0.807  |
| Tuned Random Forest        | 0.837    | 0.836  |
| Tuned Logistic Regression  | 0.837    | 0.836  |

The best performing models were **Tuned Random Forest** and **Tuned Logistic Regression**, both achieving an accuracy score of **83.7%** using **TF-IDF Vectorization**.

## Hyperparameter Tuning
The project applied **hyperparameter tuning** to improve model performance:
- **Multinomial Naive Bayes**:
  - Best model: `MultinomialNB(alpha=0.01)`
  - TF-IDF Accuracy: 0.794
- **Random Forest**:
  - Best model: `RandomForestClassifier(n_estimators=200, random_state=42)`
  - TF-IDF Accuracy: 0.816
- **Logistic Regression**:
  - Best model: `LogisticRegression(C=31.0, max_iter=150)`
  - TF-IDF Accuracy: 0.814

## Conclusion
### **Summary of Findings**:
The **Tuned Random Forest** and **Tuned Logistic Regression** models delivered the best performance, achieving an accuracy of **83.7%** using **TF-IDF vectorization**. The project demonstrated that **TF-IDF Vectorization** outperforms **CountVectorizer** for this sentiment analysis task. Additionally, hyperparameter tuning improved the performance of all models, significantly boosting accuracy and recall.

### **Recommendations**:
1. **Increase Positive Sentiment**: Use targeted campaigns to convert neutral sentiments into positive ones by enhancing customer experience.
2. **Address Negative Sentiment**: Implement a sentiment monitoring system to quickly respond to negative feedback.
3. **Engage Neutral Sentiment**: Use marketing strategies to engage neutral customers and convert their sentiment to positive.

## Future Work
- **Deploy the model**: Integrate the best-performing model into a system for **real-time sentiment analysis** of tweets.
- **Advanced Techniques**: Explore **deep learning models** such as **BERT** for further performance improvements.
- **Additional Data**: Incorporate more data sources such as reviews and forums for a broader understanding of sentiment.

## License
This project is licensed under the **MIT License** - see the LICENSE file for details.

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
