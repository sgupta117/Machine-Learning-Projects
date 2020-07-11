# Tweet-Analyzer

Web App Link : https://tweeter-analyzer-app.herokuapp.com/

Docker Container : https://hub.docker.com/repository/docker/shubham017/tweet_analyzer

# Build on  :  ![download (2)](https://user-images.githubusercontent.com/61824566/85455199-4d61fa80-b5bb-11ea-93e1-f9468a0bd1c2.jpg)

# Technology Used:

![OIP](https://user-images.githubusercontent.com/61824566/85453962-0aebee00-b5ba-11ea-83a5-e0d80286f5ea.jpg)       ![download](https://user-images.githubusercontent.com/61824566/85454285-6322f000-b5ba-11ea-8223-7544d32bf409.jpg)       ![OIP (1)](https://user-images.githubusercontent.com/61824566/85454332-71710c00-b5ba-11ea-884f-22def7bdabfb.jpg)

![download (1)](https://user-images.githubusercontent.com/61824566/85454579-b2692080-b5ba-11ea-8deb-bf474a879c4a.jpg)




This is a cool web app integrated with twitter which takes the twitter handel as as input and does :

1.Analyze the tweets of your favourite Personalities

This tool performs the following tasks :
1. Fetches the 10 most recent tweets from the given twitter handel
2. Generates a Word Cloud
3. Performs Sentiment Analysis a displays it in form of a Bar Graph

2.This tool fetches the last 100 tweets from the twitter handel & Performs the following tasks
Converts it into a DataFrame

Cleans the text
1. Analyzes Subjectivity of tweets and adds an additional column for it
2. Analyzes Polarity of tweets and adds an additional column for it
3. Analyzes Sentiments of tweets and adds an additional column for it


This respository contains all the files for end to end model building and deployment of tweet analyzer web app

Procfile : To generate command to run the app

Tweet_Analyzer.ipynb : Model building File

Twitter Data : File created after every query on the web app

Requirements.txt: Requirement file

setup.sh : predefined file for streamlite on heroku

This app is created on a tool called Streamlit which saves you from the headache of front-end devlopment ,you can install it by:
Streamlit documentation: https://docs.streamlit.io/en/latest/

pip install streamlit

& to run it on local host : streamlit run myfile.py
