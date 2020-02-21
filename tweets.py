import pandas as pd
import numpy as np
import os

# Let the program know which is the current folder and which is the folder with the data in it
PROJ_ROOT_DIR = os.getcwd()

DATA_PATH = os.path.join(PROJ_ROOT_DIR, "data")
if not os.path.isdir(DATA_PATH):  
    os.makedirs(DATA_PATH)

# Load each file in the data folder and add it to the same file
def load_twitter_data():
    """
    A function to load scraped news data from data folder
    """
    # List of files
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    
    # List of data frames
    file_list = []
    
    # Append each data frame in files to the file_list
    for filename in files:
        df = pd.read_csv(os.path.join(DATA_PATH, filename))
        file_list.append(df)
        
    # Concatenate all the news data frames
    df_full = pd.concat(file_list, join='outer').drop_duplicates().reset_index().drop(columns='index')
    
    return df_full

tweets = load_twitter_data()


# Select only the columns we are interested in
tweets_clean = tweets[['user_screen_name',  'user_display_name', 'user_reported_location', 'account_language', 'tweet_language', 'tweet_text', 'tweet_time', 'urls', 'hashtags', 'is_retweet']]

tweets_clean.head(10)

# Filter tweets and keep those that are
# 1. located in Venezuela
# 2. account language sp
# 3. tweet language sp


tweets_clean = tweets_clean[(tweets_clean.user_reported_location == 'Venezuela') | (tweets_clean.account_language == 'es') | (tweets_clean.tweet_language == 'es')].reset_index().drop(columns= ['index'])


# Take out tweets that are set in European and US location 
tweets_clean = tweets_clean[(tweets_clean.user_reported_location != 'London') & (tweets_clean.user_reported_location != 'Manhattan, NY') & (tweets_clean.user_reported_location != 'Brooklyn, NY') & (tweets_clean.user_reported_location != 'Queens, NY') & (tweets_clean.user_reported_location != 'New York, NY') & (tweets_clean.user_reported_location != 'California, USA') & (tweets_clean.user_reported_location != 'New Jersey, USA') &  (tweets_clean.user_reported_location != 'North Holland, The Netherlands') & (tweets_clean.user_reported_location != 'Atlantic City, NJ') & (tweets_clean.user_reported_location != 'Mountain View, CA') & (tweets_clean.user_reported_location != 'New York, USA') & (tweets_clean.user_reported_location != 'Canada') & (tweets_clean.user_reported_location != 'San Francisco, CA') & (tweets_clean.user_reported_location != 'Washington, USA') & (tweets_clean.user_reported_location != 'Washington, DC') & (tweets_clean.user_reported_location != 'España') & (tweets_clean.user_reported_location != 'Germany') & (tweets_clean.user_reported_location != 'Nantes, France') & (tweets_clean.user_reported_location != 'Houston, TX') & (tweets_clean.user_reported_location != 'Texas,San Antonio') & (tweets_clean.user_reported_location != 'Chicago') & (tweets_clean.user_reported_location != 'Atlanta') & (tweets_clean.user_reported_location != 'Washington,Seattle') & (tweets_clean.user_reported_location != 'Fremont, CA') & (tweets_clean.user_reported_location != 'France') & (tweets_clean.user_reported_location != 'England, United Kingdom')  & (tweets_clean.user_reported_location != 'Oregon,Portland')  & (tweets_clean.user_reported_location !='USA')  & (tweets_clean.user_reported_location != 'Florida,Orlando') & (tweets_clean.user_reported_location != 'Califor') & (tweets_clean.user_reported_location !='California,Los Angeles') & (tweets_clean.user_reported_location !='Illinois, USA') & (tweets_clean.user_reported_location !='Arizona,phoenix') & (tweets_clean.user_reported_location !='Pennsylvania,Pittsburgh') & (tweets_clean.user_reported_location !='Pennsylvania,Philadelphia') & (tweets_clean.user_reported_location !='Dallas, TX') ]


import spacy
from spacy_langdetect import LanguageDetector


# Initialize spacy with the SPANISH model
sp = spacy.load('es_core_news_sm')
sp.add_pipe(LanguageDetector(), name = 'language_detector', last = True)


def detect_language(df, content_column):
    '''
    A function to detect the language in each tweet and add to new row

    Argument: a dataframe  and content column
    Ouput: same dataframe with a new 'cleaned_content' column
    '''

    # Initialize list of languages
    spacy_language_detection = []

    # Call detect the language for each row in the data frame and append to spacy_language_detection list
    for row in df[content_column]:
        doc = sp(row)
        spacy_language_detection.append(doc._.language['language'])

    # Append language list to the data frame
    df['spacy_language_detection'] = spacy_language_detection

    return df 

tweets_clean = detect_language(df = tweets_clean, content_column = 'tweet_text')


tweets_clean.to_csv("data/tweets_clean.csv")

tweets_clean = pd.read_csv("data/tweets_clean.csv", index_col = 0)
tweets_clean.tweet_text = tweets_clean.tweet_text.astype('str')



def clean_string(text_string):
    '''
    A function to clean a string using SpaCy, removing stop-words and non-alphanumeric characters

    Argument: a text string
    Output: a cleaned string

    '''

    # Parse the text string using the english model initialized earlier
    doc = sp(text_string)
    
    # Initialize empty string
    clean = []

    # Add each token to the list if it is not a stop word, is alphanumeric, and if it's not a pronoun
    for token in doc:
        
        if token.is_alpha == False or token.is_stop == True:
            pass
        else:
            clean.append(token.lemma_)

    # Join the list into a string
    clean = " ".join(clean)

    return clean




example_spanish = tweets_clean.loc[:,'tweet_text'][15]
example_spanish_clean = clean_string(example_spanish)
print("Raw example: \n" + example_spanish)
print("\n Clean exmaple: \n" + example_spanish_clean)
doc = sp(example_spanish)
print(doc._.language['language'])


example_english = tweets_clean.loc[:,'tweet_text'][3]
example_english_clean = clean_string(example_english)
print("Raw example: \n" + example_english)
print("\n Clean exmaple: \n" + example_english_clean)
doc = sp(example_english_clean)

# Isolate tweets marked as Spanish by Twitter AND SpaCy 
spanish_tweets = tweets_clean[(tweets_clean.tweet_language == 'es') & (tweets_clean.spacy_language_detection == 'es')]
# INSPECT ABOVE FOR LANGUAGE THAT ISNT SPANISH

# Isolate tweets marked as English by Twitter AND SpaCy 
english_tweets = tweets_clean[(tweets_clean.tweet_language == 'en') & (tweets_clean.spacy_language_detection == 'en')]
# INSPECT ABOVE FOR LANGUAGE THAT ISNT ENGLISH

spanish_tweets.to_csv('data/spanish_tweets.csv')
english_tweets.to_csv('data/english_tweets.csv')

spanish_tweets.urls = spanish_tweets.urls.replace({'[]': np.nan})
spanish_tweets_url_RT  = spanish_tweets[(pd.notnull(spanish_tweets.urls) & (spanish_tweets.tweet_text.str.startswith('RT')))]
spanish_tweets_url  = spanish_tweets[pd.notnull(spanish_tweets.urls)]
spanish_tweets_no_url_no_RT  = spanish_tweets[(pd.isnull(spanish_tweets.urls) & (~spanish_tweets.tweet_text.str.startswith('RT')))]

for link in spanish_tweets.urls:
    print(link == np.nan)
