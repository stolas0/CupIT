import cleaning

from nltk.tokenize import word_tokenize


def prepare(data, flag, fltr=r'', stopwords=True, morph=True):

    # concat title and text
    data['text'] = data['title'] + ' ' + data['text']
    data.drop('title', axis=1, inplace=True)

    data = cleaning.regex(data, fltr)
    data = cleaning.lower(data)

    if stopwords or morph:
        # tokenize text
        data['text'] = data.apply(lambda row: word_tokenize(row['text']), axis=1)

        if morph:
            data = cleaning.morph(data)
        if stopwords:
            data = cleaning.stopwords(data, flag)

        # join words to text
        data['text'] = data['text'].apply(lambda row: ' '.join(row))

    return data
