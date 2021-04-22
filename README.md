# Disaster Response Pipeline Project

## Summary

The project aims to use a NLP Machine Learning Pipeline in order to detect cases and 
provide faster help when disasters occur. The data was provided by Figure Eight and is available in the 'data' folder.
The project pre-process the data containing in two '.csv' files, train a SKLearn model 
with pipeline and display the results in a interactive web app built on Flask.

## Folder organization

The project is organized as follows:

- app  
| - template  
| |- master.html            # main page of web app  
| |- go.html                # classification result page of web app  
|- run.py                   # Flask file that runs app  
  
- data  
|- disaster_categories.csv  # data with categories of disasters  
|- disaster_messages.csv    # data with text messages  
|- process_data.py          # Python script that pre-process and merge the datasets  
|- DisasterResponse.db      # database with the merged and cleaned data  
  
- models  
|- train_classifier.py      # Python script that creates a pipeline and trains the model  
|- classifier.pkl           # saved model   

- README.md

## Instalation

The code runs in Python 3.7.3 and requires the following libraries:
- Numpy (1.16.2)
- Pandas (0.24.2)
- Sklearn (0.20.3)
- Plotly (4.14.3)
- Tqdm (4.31.1)
- NLTK (3.4)
- SQLite3
- Joblib (0.16.0)
- Pickle Share (0.7.5)
- SQLAlchemy (1.3.1)
- Flask (1.1.2)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Acknowledgements & Licensing

The code provided here is free to use and it is under MIT Licensing.
