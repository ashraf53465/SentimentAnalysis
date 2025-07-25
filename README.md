# SentimentAnalysis

**IMDB Movie Review Sentiment Analysis**

This project performs sentiment analysis on IMDB movie reviews using machine learning techniques in Python. A logistic regression model is trained to classify reviews as positive or negative.

**Features**

Text preprocessing with TF-IDF vectorization Stop word removal Model training using cross-validated Logistic Regression Quick predictions on new reviews Model persistence with Pickle

**Dataset**

The dataset used is the IMDB Movie Review Dataset, containing 50,000 reviews labeled as positive or negative.
For reference the data could be sourced directly from [here](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format).

Make sure the CSV file (IMDB Dataset.csv) is available in your working directory.

**Requirements**

Python 3.x

pandas

numpy

scikit-learn

nltk

**Install dependencies using:**

bash pip install pandas numpy scikit-learn nltk Usage Clone this repository Place IMDB Dataset.csv in the root directory

**Run the script:**

bash python sentiment_analysis.py

**This will:**

Load and preprocess the data

Train and evaluate a logistic regression model

Save the trained model as saved_model.sav

Predicting Sentiment To predict your own text sentiment, modify the test list in sentiment_analysis.py:

python test = ["This movie was fantastic!"] X_test = tfidf.transform(test) saved_clf.predict(X_test) Prediction will be 'positive' or 'negative'.

**Notes**

The script uses a 50/50 train-test split.

Reviews are vectorized using TF-IDF, with stopwords from nltk.

Logistic Regression is cross-validated for robustness.

Model is saved for later use.
