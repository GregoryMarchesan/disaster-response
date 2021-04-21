import sys
import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from the parameters and merge them in a dataframe

    Parameters:
    messages_filepath (string): filepath to the file containing messages
    categories_filepath (string): filepath to the file containing categories

    Returns:
    df (dataframe): dataframe containing the loaded and merged information
    """

    df_messages = pd.read_csv(messages_filepath)
    df_messages.drop('original', inplace=True, axis=1)
    df_messages.set_index('id', inplace=True)
    
    df_categories = pd.read_csv(categories_filepath)
    df_categories.set_index('id', inplace=True)

    df = pd.concat([df_messages, df_categories], axis=1)

    return df

def clean_data(df):
    """
    Cleans and transform data from the provided dataframe, transforming the 'categories' column to a multi-column values

    Parameters:
    df (dataframe): Dataframe containing messages and categories

    Returns:
    df (dataframe): Dataframe cleaned and transformed
    """
    df['message'] = df['message'].str.lower()

    # split the 'categories' column in multi-columns one-hot encoded
    categories = df['categories'].tolist()

    df_categories = pd.DataFrame([sub.split(";") for sub in categories])
    categories = df_categories.iloc[1, :].str.split("-").tolist()
    categories = [i[0] for i in categories]
    
    df_categories.columns = categories
    for category in tqdm(categories):
        values = df_categories[category].str.split("-").tolist()
        df_categories[category] = pd.Series([i[1] for i in values]).astype(int).clip(upper=1)

    df_categories.reset_index(inplace=True, drop=True)
    df.reset_index(inplace=True, drop=True)

    df.drop('categories', axis=1, inplace=True)
    
    df_final = pd.concat([df, df_categories], axis=1)
    df_final.drop_duplicates(keep='first', inplace=True)

    return df_final


def save_data(df, database_filename):
    """
    Saves the df dataframe into a SQLite database usng the database filename provided

    Parameters:
    df (dataframe): A dataframe containing the information to be stored in the SQLite database
    database_filename (string): The SQLite database filename
    """
    conn = sqlite3.connect(database_filename)

    df.to_sql(database_filename, con = conn, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()