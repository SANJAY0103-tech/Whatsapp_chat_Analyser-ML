from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob

from heapq import nlargest
import re
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'].str.strip() == '<Media omitted>'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    return x, df_percent

def create_wordcloud(selected_user, df):
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().split()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'].str.strip() != '<Media omitted>']

    def remove_stop_words(message):
        return " ".join([word for word in message.lower().split() if word not in stop_words])

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().split()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'].str.strip() != '<Media omitted>']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common())
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.groupby('only_date').count()['message'].reset_index()

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

# Sentiment analysis using TextBlob
def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['polarity'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment'] = df['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    sentiment_counts = df['sentiment'].value_counts()
    return sentiment_counts, df

# Simple message classification based on keywords and length
def classify_message(text):
    text = text.lower()
    if re.search(r'\b(hi|hello|good morning|hey)\b', text):
        return 'Greeting'
    elif '?' in text:
        return 'Question'
    elif re.search(r'\b(bye|see you|good night|take care)\b', text):
        return 'Farewell'
    elif len(text.split()) > 7:
        return 'Informative'
    else:
        return 'Other'

def message_classification(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['category'] = df['message'].apply(classify_message)
    category_counts = df['category'].value_counts()
    return category_counts, df

# Chat summarization using frequency-based scoring of sentences
# def summarize_chats(selected_user, df, num_sentences=5):
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]

#     # Remove media and system messages
#     df = df[(df['message'].str.strip() != '<Media omitted>') & (df['user'] != 'group_notification')]

#     text = ' '.join(df['message'].dropna())
#     if not text.strip():
#         return ["No messages to summarize."]

#     # Tokenize into sentences using nltk's sent_tokenize
#     sentences = sent_tokenize(text)

#     # Prepare stop words
#     stop_words = set(stopwords.words('english'))

#     # Word frequency calculation
#     word_freq = {}
#     for sentence in sentences:
#         for word in re.findall(r'\w+', sentence.lower()):
#             if word not in stop_words:
#                 word_freq[word] = word_freq.get(word, 0) + 1

#     if not word_freq:
#         return ["Insufficient content to summarize."]

#     # Sentence scoring
#     sentence_scores = {}
#     for i, sentence in enumerate(sentences):
#         for word in re.findall(r'\w+', sentence.lower()):
#             if word in word_freq:
#                 sentence_scores[i] = sentence_scores.get(i, 0) + word_freq[word]

#     # Select top sentences
#     from heapq import nlargest
#     selected_indices = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
#     summary = [sentences[i] for i in sorted(selected_indices)]

#     return summary

# Personality insights based on avg message length and total messages
def user_personality_insights(df):
    avg_msg_len = df.groupby('user')['message'].apply(lambda x: sum(len(msg) for msg in x) / len(x))
    total_msgs = df['user'].value_counts()
    insights = []
    for user in total_msgs.index:
        personality = "Engaged"
        if avg_msg_len[user] > 40:
            personality = "Expressive"
        elif total_msgs[user] > 100:
            personality = "Highly Active"
        elif avg_msg_len[user] < 10:
            personality = "Silent Observer"
        insights.append({
            'user': user,
            'avg_msg_length': round(avg_msg_len[user], 1),
            'total_messages': total_msgs[user],
            'personality': personality
        })
    return pd.DataFrame(insights)
