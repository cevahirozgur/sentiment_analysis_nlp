
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

########################## TASK 1 - Text Preprocessing ###########################
# Step 1
df = pd.read_excel("amazon.xlsx")
df.head()

# Step 2
# a.Normalizing
df['Review'] = df['Review'].str.lower()

# b.Punctuations
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# Regular Expression
# c.Numbers
df['Review'] = df['Review'].str.replace('\d', '')
# d.Stopwords

import nltk
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# e.Rarewords
drops = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# f.Tokenization and Lemmatization

df['Review'].apply(lambda x: TextBlob(x).words).head()

df["Review"] = df["Review"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

########################### TASK 2 - Text Visualization ###########################

# Step 1 - Barplot Visualization

# a.Calculate Word Frequencies

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

# b.tf columns

tf.columns = ["words","tf"]
tf.sort_values("tf",ascending=False)

# c.Barplot

tf[tf["tf"] > 500].plot.bar(x = "words", y = "tf")
plt.show(block=True)


# Step 2 - WordCloud

# a.text
text = " ".join(i for i in df.Review)

# b. save wordcloud & generate
wordcloud = WordCloud().generate(text)

# c. visualization
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show(block=True)
#or
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)

wordcloud.to_file("wordcloud_kozmos.png")

########################### TASK 3 - Sentiment Analysis ###########################

#  Step 1
sia = SentimentIntensityAnalyzer()

# Step 2
# a. Calculate polarity_scores
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

# b. Filter by compound
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# c. Labels ("pos" , "neg")
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# d. Save Labels
df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

##################### TASK 4 - Preparation for ML / Feature Engineering ####################

df.groupby("sentiment_label")["Star"].mean()


# Step 1 : train-test datasets

train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["sentiment_label"],
                                                    random_state=25)

# Step 2 : TF-IDF Word Level

tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)


############################## TASK 5 - Modelling ###################################

# Step 1 : Lojistik Regression

log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

y_pred =log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

# Step 2 : CV Score

cross_val_score(log_model,
               x_test_tf_idf_word,
               test_y,
               scoring="accuracy",
               cv=5).mean()
#0.86


# Step 3

random_review = pd.Series(df["Review"].sample(1).values)
new_review = TfidfVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(new_review)
print(f'Review: {random_review[0]} \n Prediction: {pred}')

text  = ["terriable and poor quality product"]
new_review = TfidfVectorizer().fit(train_x).transform(text)
pred = log_model.predict(new_review)
print(f'Review: {text[0]} \n Prediction: {pred}')

text  = ["terriable quality curtain"]
new_review = TfidfVectorizer().fit(train_x).transform(text)
pred = log_model.predict(new_review)
print(f'Review: {text[0]} \n Prediction: {pred}')
# 'pos' >>>> wrong because curtain is the top1 word

############################## TASK 5 - Modelling ###################################

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()
#0.89


