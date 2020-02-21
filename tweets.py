import pandas as pd
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
