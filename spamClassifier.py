import pandas
import numpy as np
import matplotlib.pyplot as plt
import text_norm


def is_phone_number(s):
    try:
        int(s)
        if(len(s) == 11):
            return True
        return False
    except ValueError:
        return False


def calculate_prob(nfrac, ntotal):
    return nfrac/ntotal


def calculate_num_of_distinct_words(data_frame):
    words = set()
    count = 0
    for row in data_frame:
        for word in row:
            if word not in words:
                count += 1
                words.add(word)

    return count


def count_word(data_frame, w):
    count = 0
    for row in data_frame:
        for word in row:
            if word == w:
                count += 1

    return count


def count_all_words(data_frame):
    count = 0
    for row in data_frame:
        count += len(row)
    return count


def count_phonenumbers(data_frame):
    count = 0
    for row in data_frame:
        for word in row:
            if is_phone_number(word):
                count += 1

    return count


features = ['text', 'win', 'click', 'call', 'claim', 'prize', 'select', 'urgent', 'guaranteed', 'free', 'help', 'enjoy', 'flirt', 'girl', 'hot', 'wait',
            'send', 'cash', 'now', 'reply', 'date', 'area', 'award', 'gift', 'answer', 'local', 'surprise', 'chat', 'customer', 'service', 'sex', 'sexy', 'receivr', 'now', 'balance', 'gay', 'try', 'new', 'only', 'just', 'contact', 'invite', 'congratulation', 'chance', 'luck', 'miss']


features = text_norm.apply_stemming(features)

data_frame = pandas.read_csv("data/train_test.csv")
length = len(data_frame)

test_dataframe = data_frame.iloc[0: 1014]
text_norm.normalize(test_dataframe)

data_frame = data_frame.iloc[1014: length]

normalized_vectors = text_norm.normalize(data_frame)


# texts = data_frame.text.values
# mylen = np.vectorize(len)
total_num_of_docs = len(data_frame)
print("total:", total_num_of_docs)
num_of_distinct_words = calculate_num_of_distinct_words(
    normalized_vectors.text.values)

hams = data_frame.groupby('type').get_group('ham')
num_of_hams = len(hams)
ham_prob = calculate_prob(num_of_hams, total_num_of_docs)


spams = data_frame.groupby('type').get_group('spam')
num_of_spams = len(spams)
spam_prob = calculate_prob(num_of_spams, total_num_of_docs)


hams = normalized_vectors.groupby('type').get_group('ham')
num_of_ham_words = count_all_words(hams.text.values)

phone_num_ham = count_phonenumbers(hams.text.values)
phone_prob_ham = calculate_prob(
    phone_num_ham+1, num_of_ham_words+num_of_distinct_words)

ham_probs = {}
for f in features:
    num = count_word(hams.text.values, f)
    prob = calculate_prob(
        num+1, num_of_ham_words+num_of_distinct_words)
    ham_probs[f] = prob


spams = normalized_vectors.groupby('type').get_group('spam')
num_of_spam_words = count_all_words(spams.text.values)

phone_num_spam = count_phonenumbers(spams.text.values)
phone_prob_spam = calculate_prob(
    phone_num_spam+1, num_of_spam_words+num_of_distinct_words)

spam_probs = {}
for f in features:
    num = count_word(spams.text.values, f)
    prob = calculate_prob(
        num+1, num_of_spam_words+num_of_distinct_words)
    spam_probs[f] = prob

count_correct = 0
for index, row in data_frame.iterrows():
    s = row['text']
    spam_p = [spam_prob]
    ham_p = [ham_prob]
    # s = text_norm.normalize_string(s)
    seen = set()
    phone_seen = False
    for word in s:
        if word in features and word not in seen:
            seen.add(word)
            ham_p.append(ham_probs[word])
            spam_p.append(spam_probs[word])
        elif is_phone_number(word) and not(phone_seen):
            spam_p.append(phone_prob_spam)
            ham_p.append(phone_prob_ham)
            phone_seen = True

    if(len(spam_p) == 1):
        spam_p.append(calculate_prob(1, num_of_distinct_words))

    spam_seen_evidence = 1
    for p in spam_p:
        spam_seen_evidence = spam_seen_evidence * p

    ham_seen_evidence = 1
    for p in ham_p:
        ham_seen_evidence = ham_seen_evidence * p

    if(ham_seen_evidence > spam_seen_evidence):
        predict = 'ham'
    else:
        predict = 'spam'

    if(predict == row['type']):
        count_correct += 1
print("correctness: ", str(count_correct*100/total_num_of_docs), "%")

count_correct = 0
for index, row in test_dataframe.iterrows():
    s = row['text']
    spam_p = [spam_prob]
    ham_p = [ham_prob]
    # s = text_norm.normalize_string(s)
    seen = set()
    phone_seen = False
    for word in s:
        if word in features and word not in seen:
            seen.add(word)
            ham_p.append(ham_probs[word])
            spam_p.append(spam_probs[word])
        elif is_phone_number(word) and not(phone_seen):
            spam_p.append(phone_prob_spam)
            ham_p.append(phone_prob_ham)
            phone_seen = True

    if(len(spam_p) == 1):
        spam_p.append(calculate_prob(1, num_of_distinct_words))

    spam_seen_evidence = 1
    for p in spam_p:
        spam_seen_evidence = spam_seen_evidence * p

    ham_seen_evidence = 1
    for p in ham_p:
        ham_seen_evidence = ham_seen_evidence * p

    if(ham_seen_evidence > spam_seen_evidence):
        predict = 'ham'
    else:
        predict = 'spam'

    if(predict == row['type']):
        count_correct += 1

    if predict != row['type']:
        print("text: ", row['text'])

print(count_correct)
print("correctness: ", str(count_correct*100/len(test_dataframe)), "%")


data_frame = pandas.read_csv("data/evaluate.csv")
data_frame = text_norm.normalize(data_frame)
ids = []
types = []

for index, row in data_frame.iterrows():
    ids.append(int(index)+1)
    s = row['text']
    spam_p = [spam_prob]
    ham_p = [ham_prob]
    # s = text_norm.normalize_string(s)
    seen = set()
    phone_seen = False
    for word in s:
        if word in features and word not in seen:
            seen.add(word)
            ham_p.append(ham_probs[word])
            spam_p.append(spam_probs[word])
        elif is_phone_number(word) and not(phone_seen):
            spam_p.append(phone_prob_spam)
            ham_p.append(phone_prob_ham)
            phone_seen = True

    if(len(spam_p) == 1):
        spam_p.append(calculate_prob(1, num_of_distinct_words))

    spam_seen_evidence = 1
    for p in spam_p:
        spam_seen_evidence = spam_seen_evidence * p

    ham_seen_evidence = 1
    for p in ham_p:
        ham_seen_evidence = ham_seen_evidence * p

    if(ham_seen_evidence > spam_seen_evidence):
        predict = 'ham'
    else:
        predict = 'spam'

    types.append(predict)


dc = pandas.DataFrame({'id': ids, 'type': types})


dc.to_csv("output.csv", sep=',', index=False)
