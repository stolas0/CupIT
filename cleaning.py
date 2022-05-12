import re
import pymorphy2


# delete unnecessary symbols
def regex(data, fltr):

    reg = re.compile(fltr)
    data['text'].replace(regex=True, inplace=True, to_replace=reg, value=r' ')

    return data


# to lower
def lower(data):

    data['text'] = data['text'].str.lower()

    return data


# remove stop_words
def stopwords(data, flag):

    with open('../CupIT2019/' + flag + '/stopwords.txt', 'r', encoding='urf-8') as file:
        stop_words = [line.strip() for line in file]

    data['text'] = data['text'].apply(lambda row: [word for word in row if word not in stop_words])

    return data


# morphological analysis
def morph(data):

    MORPH = pymorphy2.MorphAnalyzer()
    data['text'] = data['text'].apply(lambda row: [MORPH.parse(word)[0].normal_form for word in row])

    return data
