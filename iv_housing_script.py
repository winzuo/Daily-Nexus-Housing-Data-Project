! pip install vaderSentiment
! pip install transformers
from google.colab import drive
drive.mount('/content/drive')
import nltk
nltk.download([
    "names",
    "stopwords",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
    "wordnet"
])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, twitter_samples
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re
import string
import random
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Import data below
our_data = pd.read_csv("/content/rating_revised.csv")
our_data

# This chunk can be used to break our reviews down into the components that we
# can then run VADER on.
def remove_noise(review, stop_words=()):
    cleaned_tokens = []
    for token, tag in pos_tag(review):
      # Remove unnecesassry and irrelevant characters
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

      # Provide tag for part of speech
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"
      # Set words to their base forms
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        token = ''.join(char for char in token if char not in string.punctuation)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# Extract words from the cleaned tokens list
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# Extract reviews from the list
def get_reviews(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# Process the reviews so that we can use it for sentiment analysis
def process_reviews(reviews):
    stop_words = stopwords.words('english')

    cleaned_tokens_list = []

    for review in reviews:
        review_tokens = word_tokenize(review)
        cleaned_tokens = remove_noise(review_tokens, stop_words)
        cleaned_tokens_list.append(cleaned_tokens)

    all_words = list(get_all_words(cleaned_tokens_list))
    tokens_for_model = list(get_reviews(cleaned_tokens_list))
    return cleaned_tokens_list

# Function to apply VADER to reviews to conduct sentiment analysis
def analyze_sentiment(reviews):
  sia = SentimentIntensityAnalyzer()
  sentiment_scores = sia.polarity_scores(reviews)
  return sentiment_scores


sample_sentence = ["This is a review of a really great product that I enjoy! I also think it's the worst", "The management is horrible. Students are being used as bait and are ashamed to work for a company with a bad rep.This guy Richard who's the manager knows he can take your deposit using tricks.Email is the way we were told to approach the deposit claim and after a professional email we get a one sentence response. He approaches you from a pedestal trying to explain his immoral practices in the most absurd ways doesn't even bother to fully explain.Falsified ridiculous extra charges not worth disputing because of your time patience and mental health with someone who won't cooperate in a manner besides taking issues to court.Covid fees which were not in the contract.Overcharged hauling fees.Ignored fix it replies.Year after year playa IV spits lies to oblivious international students ripping them off on the earlier mentioned deposits.After many posts on Facebook suggested that students should complain on social media, PLAYA IV started replying( on social media) with someone who has knowledge in PR as seen below trying to ease the extensive damage to their reputation.PsI forgot to mention there was a building for demolition right next to ours and PLAYA IV forgot to mention that it would be super sour for the whole year. Especially with online school. No compassion.Greed for money."]
result = process_reviews(sample_sentence)

# Initilize dataframe
data = {'Review': [], 'Sentiment Scores': [], 'Sentiment Category': []}

for tokens in result:
  review_text = ' '.join(tokens)
  sentiment_scores = analyze_sentiment(review_text)

  data['Review'].append(' '.join(tokens))
  data['Sentiment Scores'].append(sentiment_scores['compound'])

  # Determine sentiment category for each individual score
  if sentiment_scores['compound'] >= 0.05:
      data['Sentiment Category'].append("Positive")
  elif sentiment_scores['compound'] <= -0.05:
      data['Sentiment Category'].append("Negative")
  else:
      data['Sentiment Category'].append("Neutral")

# Create DataFrame
df = pd.DataFrame(data)



print(df)

def shorten_reviews(dataframe):
    # Function to tokenize, limit tokens to 406, and detokenize
    def process_sentence(sentence):
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > 414:
            tokens = tokens[:414]  # Truncate to the first 406 tokens
        shortened_sentence = ' '.join(tokens)
        return shortened_sentence

    # Shorten each sentence and store in a list
    shortened_reviews = [process_sentence(review) for review in dataframe['REVIEW']]

    result_df = pd.DataFrame({'Date': dataframe['DATE'], 'Truncated Reviews': shortened_reviews})

    # Create a DataFrame from the shortened data
    return result_df

# Call the function and get the DataFrame
result_df = shorten_reviews(our_data)

# Print the DataFrame
print(result_df)

def analyze_sentiment_transformer(dataframe):


    dfs = []

    for index, row in dataframe.iterrows():
        try:
            sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            sentiment_output = sentiment_pipeline(row['Truncated Reviews'])

            # Create a DataFrame with reviews, labels, and scores
            df_data = {'Date': row['Date'],
                       'Review': row['Truncated Reviews'],
                       'Label': [item['label'] for item in sentiment_output],
                       'Score': [item['score'] for item in sentiment_output]}

            df = pd.DataFrame(df_data)
            dfs.append(df)
        except RuntimeError as e:
            if "The size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                print(f"RuntimeError: {e}. Skipping to the next iteration.")
                continue
            else:
                raise  # Re-raise the exception if it's not the expected one

    final_df = pd.concat(dfs, ignore_index=True)

    return final_df

sentiment_results = analyze_sentiment_transformer(result_df)

sentiment_results.to_csv("/content/sentiment_analyzed2.csv")

sentiment_results = pd.read_csv("/content/sentiment_analyzed2.csv").iloc[:, 1:]

sentiment_results['Month Year'] = pd.to_datetime(sentiment_results['Date'], format = '%m/%d/%Y').dt.strftime('%B %Y')
sentiment_results.to_csv("/content/sentiment_analyzed_monthyear2.csv")

positive_reviews = sentiment_results[sentiment_results["Label"] == "POSITIVE"]
negative_reviews = sentiment_results[sentiment_results["Label"] == "NEGATIVE"]

print("Positive Reviews Count:", positive_reviews.shape[0], "\nNegative Reviews Count:", negative_reviews.shape[0])

def find_ngrams(ngram_length = 2, in_reviews = sentiment_results.iloc[:, 2], top_n = 5):
  processed_text = process_reviews(in_reviews)
  flat_processed_text = [item for sublist in processed_text for item in sublist]

  if ngram_length == 1:
    fd = nltk.FreqDist(flat_processed_text)
    the_word = fd.most_common(top_n)

  if ngram_length == 2:
    bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(flat_processed_text)
    the_word = bigram_finder.ngram_fd.most_common(top_n)

  if ngram_length == 3:
    trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(flat_processed_text)
    the_word = trigram_finder.ngram_fd.most_common(top_n)
  if ngram_length == 4:
    quadgram_finder = nltk.collocations.QuadgramCollocationFinder.from_words(flat_processed_text)
    the_word = quadgram_finder.ngram_fd.most_common(top_n)

  return the_word

find_ngrams(ngram_length=3, in_reviews=negative_reviews.iloc[:,1])

pos_word_freq = dict(find_ngrams(ngram_length = 1, in_reviews = positive_reviews.iloc[:,1], top_n = 250))

neg_word_freq = dict(find_ngrams(ngram_length = 1, in_reviews = negative_reviews.iloc[:, 1], top_n = 250))

def get_words_for_cloud(dictionary, out_file_name):
  sorted_words = sorted(dictionary.items(), key = lambda x: x[1], reverse = True)

  with open(out_file_name, 'w') as file:
    for word, freq in sorted_words:
      file.write(f"{word} " * freq)

  file.close()

get_words_for_cloud(pos_word_freq, "positive_words_for_cloud.txt")

get_words_for_cloud(neg_word_freq, "negative_words_for_cloud.txt")

find_ngrams(ngram_length = 1, in_reviews = negative_reviews.iloc[:,1], top_n = 50)

find_ngrams(ngram_length = 2, in_reviews = positive_reviews.iloc[:,1], top_n = 25)

find_ngrams(ngram_length = 2, in_reviews = negative_reviews.iloc[:,1], top_n = 50)

find_ngrams(ngram_length = 3, in_reviews = positive_reviews.iloc[:,1], top_n = 25)

find_ngrams(ngram_length = 3, in_reviews = negative_reviews.iloc[:,1], top_n = 25)

find_ngrams(ngram_length = 4, in_reviews = positive_reviews.iloc[:,1], top_n = 25)

find_ngrams(ngram_length = 4, in_reviews = negative_reviews.iloc[:,1], top_n = 25)

# This script can be used to break the data down into smaller groups for things
# like word frequency, bigrams, trigrams, etc.
if __name__ == "__main__":
    stop_words = stopwords.words('english')
    reviews = ["This is a review of a really great product that I enjoy! I also think it's the worst", "this is yet another review"]

    cleaned_tokens_list = []

    for review in reviews:
        review_tokens = word_tokenize(review)
        cleaned_tokens = remove_noise(review_tokens, stop_words)
        cleaned_tokens_list.append(cleaned_tokens)

    all_words = list(get_all_words(cleaned_tokens_list))
    tokens_for_model = list(get_reviews(cleaned_tokens_list))

    print(cleaned_tokens_list)

def unprocessed_scoops(review_in):
  data = {'Review': [], 'Sentiment Scores': [], 'Sentiment Category': []}

  for review in review_in:
    sentiment_scores = analyze_sentiment(review)
    data['Review'].append(review)
    data['Sentiment Scores'].append(sentiment_scores['compound'])

    if sentiment_scores['compound'] >= 0.05:
      data['Sentiment Category'].append("Positive")
    elif sentiment_scores['compound'] <= -0.05:
      data['Sentiment Category'].append("Negative")
    else:
      data['Sentiment Category'].append("Neutral")

  df = pd.DataFrame(data)

  return df


unprocessed_scoops(our_data)

# Script to run for sentiment Analysis:

def give_me_the_scoop(review_in):
  housing_reviews = process_reviews(review_in)

  data = {'Review': [], 'Sentiment Scores': [], 'Sentiment Category': []}

  for tokens in housing_reviews:
    review_text = ' '.join(tokens)
    sentiment_scores = analyze_sentiment(review_text)

    data['Review'].append(' '.join(tokens))
    data['Sentiment Scores'].append(sentiment_scores['compound'])

    if sentiment_scores['compound'] >= 0.05:
      data['Sentiment Category'].append("Positive")
    elif sentiment_scores['compound'] <= -0.05:
      data['Sentiment Category'].append("Negative")
    else:
      data['Sentiment Category'].append("Neutral")

  df = pd.DataFrame(data)

  return df


give_me_the_scoop(our_data)

result_df