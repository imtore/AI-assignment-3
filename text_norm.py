import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')


def convert_lowercase(s):
    return s.lower()


def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))


def remove_whitespace(s):
    return s.strip()


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [i for i in tokens if not i in stop_words]


def apply_stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(i) for i in tokens]


def normalize(sen_dataframe):
    for index, row in sen_dataframe.iterrows():
        row['text'] = convert_lowercase(row['text'])
        row['text'] = remove_punctuation(row['text'])
        row['text'] = remove_whitespace(row['text'])
        row['text'] = word_tokenize(row['text'])
        row['text'] = remove_stopwords(row['text'])
        row['text'] = apply_stemming(row['text'])

    return sen_dataframe


# spam_len = mylen(spams.text.values)


# ham_len = mylen(hams.text.values)


# plt.plot(spam_len, marker='x', color='r', ls='')

# # plt.title('price per duration')
# # plt.xlabel('duration')
# # plt.ylabel('price')
# plt.grid()
# plt.show()

# plt.plot(ham_len, marker='+', color='g', ls='')
# plt.show()

# texts = data_frame.text.values
# mylen = np.vectorize(len)
