import sys
import numpy as np 
import pandas as pd 
from tqdm import tqdm 

import sqlite3
import pickle
import joblib

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    """
    Loads the database from a SQLite file into the script and split it in X, y and category names

    Parameters:
    database_filename (string): name of the database

    Returns:
    X (numpy array): Array containing the X values (messages)
    y (numpy array): Array containing the y values (one hot enconding of categories)
    category_names (list): List containing the category names in the database
    """
    conn = sqlite3.connect(database_filepath)

    # get a cursor
    cur = conn.cursor()

    # create the test table including project_id as a primary key
    df = pd.read_sql("SELECT * FROM '{}'".format(database_filepath), con=conn)

    conn.commit()
    conn.close()

    # genre_one_hot = pd.get_dummies(df["genre"])

    # X = pd.concat([df["message"], genre_one_hot], axis=1).values
    X = df["message"].values
    y = df.drop(["message", "genre"], axis=1).values

    category_names = df.drop(["message", "genre"], axis=1).columns

    return X, y, category_names


def tokenize(text):
    """
    Tokenize the text to further process in the ML algorithm

    Parameters:
    text (string): the text from each message

    Returns:
    clean_tokens (list): The list of tokens processed
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Using grid search, builds the model to classify the messages

    Returns:
    model (): The trained model over the data
    """

    text_pipeline = Pipeline([
                              ('vect', CountVectorizer(tokenizer=tokenize)),
                              ('tfidf', TfidfTransformer())
                            ])

    pipeline = Pipeline([
        ('text_pipeline', text_pipeline),
        ('clf', RandomForestClassifier())
    ])

    parameters = {
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'text_pipeline__vect__max_features': (None, 5000, 10000),
        'text_pipeline__tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the trained model

    Parameters:
    model (sklearn model): the trained model
    X_test (numpy array): An array with X values to test the model
    Y_test (numpy array): An array with Y values to test the model
    category_names (list): Names of the categories

    """
    Y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        precision = precision_score(Y_test[i], Y_pred[i])
        recall = recall_score(Y_test[i], Y_pred[i])
        f1 = f1_score(Y_test[i], Y_pred[i])

        print("", category)
        print("Category: {} | F1-Score: {:.2f} % | Precision: {:.2f} % | Recall: {:.2f} %".format(category, 
                                                                                                  f1*100, 
                                                                                                  precision*100, 
                                                                                                  recall*100))


def save_model(model, model_filepath):
    """
    Saves the trained model into a pickle file

    Parameters: 
    model (sklearn model): The trained model
    model_filepath (string): The path to save the model
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()