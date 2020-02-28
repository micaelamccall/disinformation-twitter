import pandas as pd
import numpy as np
import os

# Let the program know which is the current folder and which is the folder with the data in it
PROJ_ROOT_DIR = os.getcwd()

DATA_PATH = os.path.join(PROJ_ROOT_DIR, "src_data")
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
eng = spacy.load('en_core_web_sm')

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


tweets_clean.to_csv("proj_data/tweets_clean.csv", encoding='utf-8-sig')

tweets_clean = pd.read_csv("csv/tweets_clean.csv", index_col = 0, encoding='utf-8-sig')
tweets_clean.tweet_text = tweets_clean.tweet_text.astype('str')

# Isolate tweets marked as Spanish by Twitter AND SpaCy 
spanish_tweets = tweets_clean[(tweets_clean.tweet_language == 'es') & (tweets_clean.spacy_language_detection == 'es')]


# Isolate tweets marked as English by Twitter AND SpaCy 
english_tweets = tweets_clean[(tweets_clean.tweet_language == 'en') & (tweets_clean.spacy_language_detection == 'en')]



def clean_string(text_string, language):
    '''
    A function to clean a string using SpaCy, removing stop-words and non-alphanumeric characters

    Argument: a text string and a language ('English' or 'Spanish')
    Output: a cleaned string

    '''
    if language == 'Spanish':
    # Parse the text string using the english model initialized earlier
        doc = sp(text_string)
    elif language == 'English':
        doc = eng(text_string)
    
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


def clean_content(df, content_column, language):
    '''
    A function to clean all the strings in a whole of a corpus

    Argument: a dataframe, the name of the column with the content, and a language ('Spanish' or 'English')
    Ouput: same dataframe with a new cleaned content column
    '''

    # Initialize list of cleaned content strings
    clean_content= []

    # Call clean_string() for each row in the data frame and append to clean_content list
    for row in df[content_column]:
        clean_content.append(clean_string(row, language))

    # Append clean_content list to the data frame
    df['lemmatized_tweet_text'] = clean_content

    return df 

spanish_tweets = clean_content(spanish_tweets, 'tweet_text', 'Spanish')

english_tweets = clean_content(english_tweets, 'tweet_text', 'English')

spanish_tweets.to_csv('csv/spanish_tweets.csv', encoding='utf-8-sig')
english_tweets.to_csv('csv/english_tweets.csv', encoding='utf-8-sig')

spanish_tweets = pd.read_csv('csv/spanish_tweets.csv', encoding='utf-8-sig', index_col = 0).reset_index().drop(columns='index')
english_tweets = pd.read_csv('csv/english_tweets.csv', encoding='utf-8-sig', index_col = 0).reset_index().drop(columns='index')

from sklearn.feature_extraction.text import CountVectorizer

# only words that appear in more than 5 tweets (a way to decrease the size of the vocab)
word_vectorizer = CountVectorizer(encoding='utf-8-sig', analyzer='word', min_df=5, ngram_range=(1,1))
# create matrix where each column is a word and each row is a count in each tweet
word_count_sm = word_vectorizer.fit_transform(spanish_tweets['lemmatized_tweet_text'])
words = word_vectorizer.get_feature_names()
# word_count_df = pd.DataFrame(word_count_sm.todense(), columns=words).merge(spanish_tweets[['tweet_text']].reset_index().drop(columns='index'), how='outer', left_index= True, right_index=True)

# sum the count of each word over all Tweets
word_count_total = word_count_sm.sum(axis=0)
word_count_total_df = pd.DataFrame(word_count_total, columns = words)
word_count_total_df.to_csv('xlsx/spanish_word_count_total.csv')



# only phrases that appear in more than 5 tweets (a way to decrease the size of the vocab)
phrase_vectorizer = CountVectorizer(encoding='utf-8-sig', analyzer='word', min_df=5, ngram_range=(5,7))
# create matrix where each column is a phrase and each row is a count in each tweet
phrase_count_sm = phrase_vectorizer.fit_transform(spanish_tweets['lemmatized_tweet_text'])
phrases = phrase_vectorizer.get_feature_names()
# sum the count of each phrase over all Tweets
phrase_count_total = phrase_count_sm.sum(axis=0)
phrase_count_total_df = pd.DataFrame(phrase_count_total, columns = phrases)
# phrase_count_df = pd.DataFrame(phrase_count_sm.todense(), columns= phrases).merge(spanish_tweets[['tweet_text']].reset_index().drop(columns='index'), how='outer', left_index= True, right_index=True)
phrase_count_total_df.to_csv('xlsx/spanish_phrase_34words_count_total.csv', encoding='utf-8-sig')
phrase_count_total_df.to_csv('xlsx/spanish_phrase_57words_count_total.csv', encoding='utf-8-sig')


## ENGLISH ##
# create matrix where each column is a word and each row is a count in each tweet
word_count_sm = word_vectorizer.fit_transform(english_tweets['lemmatized_tweet_text'])
words = word_vectorizer.get_feature_names()
# word_count_df = pd.DataFrame(word_count_sm.todense(), columns=words).merge(spanish_tweets[['tweet_text']].reset_index().drop(columns='index'), how='outer', left_index= True, right_index=True)


# sum the count of each word over all Tweets
word_count_total = word_count_sm.sum(axis=0)
word_count_total_df = pd.DataFrame(word_count_total, columns = words)
word_count_total_df.to_csv('xlsx/english_word_count_total.csv')



# only phrases that appear in more than 5 tweets (a way to decrease the size of the vocab)
phrase_vectorizer = CountVectorizer(encoding='utf-8-sig', analyzer='word', min_df=5, ngram_range=(5,7))
# create matrix where each column is a phrase and each row is a count in each tweet
phrase_count_sm = phrase_vectorizer.fit_transform(english_tweets['lemmatized_tweet_text'])
phrases = phrase_vectorizer.get_feature_names()
# sum the count of each phrase over all Tweets
phrase_count_total = phrase_count_sm.sum(axis=0)
phrase_count_total_df = pd.DataFrame(phrase_count_total, columns = phrases)
# phrase_count_df = pd.DataFrame(phrase_count_sm.todense(), columns= phrases).merge(spanish_tweets[['tweet_text']].reset_index().drop(columns='index'), how='outer', left_index= True, right_index=True)
phrase_count_total_df.to_csv('xlsx/english_phrase_34words_count_total.csv', encoding='utf-8-sig')
phrase_count_total_df.to_csv('xlsx/english_phrase_57words_count_total.csv', encoding='utf-8-sig')


# Isolate the tweet that mention Venezuela or Venezolano/a
spanish_tweets_venezuela = spanish_tweets[spanish_tweets.tweet_text.str.contains("venez")]

# Drop duplicates in the tweet text column to see how many are 
spanish_tweets_no_content_copy = spanish_tweets.tweet_text.drop_duplicates()

# Remove parens and filter hashtags
hashtags = []
for hashtag in spanish_tweets.hashtags:
    if type(hashtag) == float:
        hashtags.append('None')
    elif len(hashtag) < 3:
        hashtags.append('None')
    else:
        hashtags.append(hashtag[1:-1])
# Add filtered hashtags back to data frame
spanish_tweets.hashtags = hashtags

# Spanish hashtag count and save 
hashtag_vectorizer = CountVectorizer(encoding='utf-8-sig', analyzer='word', ngram_range=(1,1))
hashtag_count_sm = hashtag_vectorizer.fit_transform(spanish_tweets['hashtags'])
hashtags = hashtag_vectorizer.get_feature_names()
hashtag_total = hashtag_count_sm.sum(axis = 0)
hashtag_count_df = pd.DataFrame(hashtag_total, columns= hashtags)
hashtag_count_df.to_csv('xlsx/spanish_hashtag_count_total.csv', encoding='utf-8-sig')


# Tweets without link
spanish_tweets.urls = spanish_tweets.urls.replace({'[]': np.nan})




tweets_with_link = 0
tweets_wout_link = 0

for url in spanish_tweets.urls:
    if pd.isnull(url) == True:
        tweets_with_link +=1
    else:
        tweets_wout_link += 1



spanish_tweets_url_RT  = spanish_tweets[(pd.notnull(spanish_tweets.urls) & (spanish_tweets.tweet_text.str.startswith('RT')))]
spanish_tweets_url  = spanish_tweets[pd.notnull(spanish_tweets.urls)]
spanish_tweets_url.to_csv('xlsx/spanish_tweets_url.csv', encoding='utf-8-sig')
spanish_tweets_url_at = spanish_tweets_url[spanish_tweets_url.tweet_text.str.contains('@')]
spanish_tweets_url_at.to_csv('xlsx/spanish_tweets_url_@.csv', encoding='utf-8-sig')
spanish_tweets_url_at_rt = spanish_tweets_url_at[spanish_tweets_url_at.tweet_text.str.startswith('RT')]
spanish_tweets_url_at_rt.to_csv('xlsx/spanish_tweets_url_@_rt.csv', encoding='utf-8-sig')


tweet_date = []
for date in spanish_tweets.tweet_time:
    tweet_date.append(str(date)[:-5].strip())

spanish_tweets['tweet_date'] = tweet_date

fourteen = 0
fifteen = 0
sixteen = 0
seventeen = 0
eighteen = 0

for date in spanish_tweets['tweet_date']:
    if date.endswith('14'):
        fourteen += 1
    elif date.endswith('15'):
        fifteen +=1
    elif date.endswith('16'):
        sixteen += 1
    elif date.endswith('17'):
        seventeen += 1
    elif date.endswith('18'):
        eighteen +=1

tweets_from_nineteen = spanish_tweets[spanish_tweets['tweet_date'].str.endswith('19')]

spanish_june_nineteen = spanish_tweets[(spanish_tweets['tweet_date'].str.startswith('6') & (spanish_tweets['tweet_date'].str.endswith('19')))]




spanish_tweets_no_url_no_RT  = spanish_tweets[(pd.isnull(spanish_tweets.urls) & (~spanish_tweets.tweet_text.str.startswith('RT')))]

for link in spanish_tweets.urls:
    print(link == np.nan)
