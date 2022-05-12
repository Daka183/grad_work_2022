import pandas as pd
import re
from tqdm import tqdm
import nltk
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def text_cleaning (df):
    df = df.replace(".", "")
    df = df.replace("[", "")
    df = df.replace(",", "")
    df = df.replace("]", "")
    df = df.replace("(", "")
    df = df.replace(")", "")
    df = df.replace("\"", "")
    df = df.replace("-", "")
    df = df.replace("=", "")

    for i in tqdm(range(len(df))):
        sentence = df[i]

        # normalization
        sentence = re.sub(r'https?://\S+|www\.\S+', r'', sentence) # URLS
        sentence = re.sub(r'@\w+','', sentence) # mentions
        sentence = re.sub(r'#\w+','', sentence) # hash
        sentence = sentence.lower()
        sentence = re.sub(r'\d+', '', sentence).strip() # number
        sentence = re.sub(r"[^\w\s\d]","", sentence) # puctuations
        sentence = re.sub(r"\s+"," ", sentence).strip() # space
        sentence = sentence.replace("\n", " ")
        sentence = re.sub("\S*\d\S*", "", sentence).strip()
        sentence = re.sub('[^A-Za-z]+', ' ', sentence)

        # stop words removing
        STOPWORDS = set(stopwords.words('english'))
        sentence = ' '.join([e for e in sentence.split() if e not in STOPWORDS])
        sentence = sentence.strip()

        # tokenization
        sentence = nltk.word_tokenize(sentence)

        df[i] = sentence

    return df


def lemmatization (df):
    #nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

    for i in tqdm(range(len(df))):
        df[i] = [lemmatizer.lemmatize(word) for word in df[i]]
    
    return df


def vectorization (X_train, X_test):
    for i in range(len(X_train)):
        X_train[i] =' '.join([str(word) for word in X_train[i]])

    for i in range(len(X_test)):
        X_test[i] =' '.join([str(word) for word in X_test[i]])

    tf_idf_vect = TfidfVectorizer()
    tf_idf_vect.fit(X_train)

    X = tf_idf_vect.transform(X_train)
    X_train = pd.DataFrame(data=X.toarray(), columns=tf_idf_vect.get_feature_names())

    X = tf_idf_vect.transform(X_test)
    X_test = pd.DataFrame(data=X.toarray(), columns=tf_idf_vect.get_feature_names())

    return [X_train, X_test]
    


