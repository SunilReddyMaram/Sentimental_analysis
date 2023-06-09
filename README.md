# Sentimental_analysis

**Introduction**
Sentiment Analysis is a process of identifying the feeling or emotions based on the text or speech data provided. Natural Language processing is used to process the text or speech data to identify the unique characteristics of the text and classify them into positive, negative or neutral sentiment. There are many python packages like NLTK, TEXTBLOB, FLAIR, SPACY, PYTORCH etc that can be used for performing sentiment analysis on the text data.
The applications of sentiment analysis are numerous and are employed by many businesses. Some of the applications are
● Social media monitoring.
● Customer support ticket analysis.
● Brand monitoring and reputation management.
● Listen to voice of the customer (VoC).
● Listen to voice of the employee.
● Product analysis.
● Market research and competitive research.
● Movie Review and Recommendation Systems.

**Problem Statement** 
Review dataset is a common example of sentiments provided by the customer, clients on the products and services respectively. We choose a Movie Review Dataset from IMDB to analyse the sentiments. The aim of the project is to accurately classify the review as a positive or a negative review based on the training dataset split from the whole dataset.
**Data Acquisition.**
● The source of the data is
https://ai.stanford.edu/~amaas/data/sentiment/.
● The data is provided in form of a test and train with equal positive and negative review data.
● The team compiled the dataset into a single csv file named IMDB Dataset.csv.
Sentimental Analysis packages
The NLP packages are imported from NLTK. Initial attempts were made to install Pytorch and SpaCy but the team faced issues with currently installed softwares and was unable to access these packages. The classification & accuracy packages are imported from SKLEARN. The following packages are used in building the sentiment analysis model.

**Text Data cleaning** 
There is no missing data and the data is exactly symmetrical with half negative and positive reviews respectively. However, the review is cleaned using the packages mentioned above in the following steps.
● Import Data,
● Null check,
● Target Variable Check,
● Declaring a Tokenizer,
● DeNoising the text (Removing HTML data , special characters),
● Stemming the text data (removing the unnecessary suffixes),
● Removing Stopwords,
● Obtain clean data.

**Feature transformation:**
The processed data is a clean text data that needs to be transformed into features and to transform the text reviews into numerical values two kinds of vectorizations are used and they are CountVectorizer (Bag of Words) and TFIDFVectorizer ( Term Frequency -Inverse Document frequency). To convert the text into metric based on the frequency of the words both these vectorizations methods are used based on the rule that frequently used words carry less importance in deriving the sentiment of the statement.
The data is split into test and training data for the features and vectorization methods are applied. The target variable is also divided into test and train split. The partition to train to test ratio is 80% to 20%. The data is featured to train the classification models.

**Model Selection and Results**
Three models of classification are used to model the review dataset. Logistic Regression, SVM(Support Vector Machine) and Naive Bayes are used to classify the models. Naive Bayes in general is a good model to carry to text classifications. Our systems are not capable of Neural Network classification with 50,000 records of text data so we had to stop proceeding with the Neural Network classifier.

**The accuracy results of the models are as follows:**

**Logistic Regression:** 
Bag of Words - 75%
TFIDFVectorizer -75%.

**SVM:**
Bag of Words - 58%
TFIDFVectorizer - 51%.

**Naive Bayes:**
Bag of Words - 75%
TFIDFVectorizer -75%.

**Logistic Regression and Naive Bayes are able to classify with better accuracy for both vectorization methods.**
 
