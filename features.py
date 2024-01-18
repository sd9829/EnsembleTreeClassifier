import re

DUTCH_COMMON_WORDS = ['ik', 'je', 'het', 'de', 'dat', 'een', 'niet', \
                      'en', 'wat', 'van', 'ze', 'op', 'te', 'hij', 'zijn', 'er', \
                      'maar', 'die', 'heb', 'voor', 'met', 'als', 'ben', 'mijn', 'u', \
                      'dit', 'aan', 'om', 'hier', 'naar', 'dan', 'jij', 'zo', 'weet', \
                      'ja', 'kan', 'geen', 'nog', 'wel', 'wil', 'moet', 'goed', 'hem', \
                      'hebben', 'nee', 'heeft', 'waar', 'nu', 'hoe', 'ga', 'kom', 'uit', \
                      'al', 'jullie', 'zal', 'bij', 'ons', 'gaat', 'hebt', 'meer', \
                      'waarom', 'iets', 'deze', 'laat', 'doe', 'm', 'moeten', 'wie', \
                      'jou', 'alles', 'denk', 'kunnen', 'eens', 'echt', 'weg', \
                      'terug', 'laten', 'mee', 'hou', 'komt', 'toch', 'zien', 'oké', 'alleen', 'nou', 'dus', 'nooit',
                      'niets', 'zei', \
                      'misschien', 'kijk', 'iemand', 'komen', 'tot', 'veel', \
                      'worden', 'onze', 'mensen', 'zeg', 'leven', 'zeggen', 'weer', \
                      'gewoon', 'nodig', 'jouw', 'vrouw', 'geld', 'wij', 'twee', 'tijd', 'tegen', 'uw', \
                      'toen', 'zit', 'net', 'weten', 'heel', 'maken', 'wordt', \
                      'dood', 'mag', 'altijd', 'af', 'wacht', 'geef', 'z', 'lk', \
                      'dag', 'omdat', 'zeker', 'zie', 'allemaal', 'gedaan', 'oh', \
                      'dank', 'huis', 'hé', 'zij', 'jaar', 'vader', 'doet', 'zoals', \
                      'hun']

ENGLISH_COMMON_WORDS = ['about', 'all', 'also', 'and', 'as', 'at', 'be', \
                        'because', 'but', 'by', 'can', 'come', 'could', 'day', 'do', 'even', \
                        'find', 'first', 'for', 'from', 'get', 'give', 'go', 'have', 'he', \
                        'her', 'here', 'him', 'his', 'how', 'I', 'if', 'in', 'into', 'it', \
                        'its', 'just', 'know', 'like', 'look', 'make', 'man', 'many', 'me', \
                        'more', 'my', 'new', 'no', 'not', 'now', 'of', 'on', 'one', 'only', \
                        'or', 'other', 'our', 'out', 'people', 'say', 'see', 'she', 'so', 'some', \
                        'take', 'tell', 'than', 'that', 'their', 'them', 'then', 'there', \
                        'these', 'they', 'thing', 'think', 'this', 'those', 'time', 'to', \
                        'two', 'up', 'use', 'very', 'want', 'way', 'we', 'well', 'what', \
                        'when', 'which', 'who', 'will', 'with', 'would', 'year', 'you', 'your']

DUTCH_DIPTHONGS = ['ae', 'ei', 'au', 'ai', 'eu', 'ie', 'oe', 'ou' \
                                                             'ui', 'aai', 'oe', 'ooi', 'eeu', 'ieu']

ENGLISH_ARTICLES = ['an', 'the']


def isDutch(statement):
    words = statement.split()
    for word in words:
        if word.lower() in DUTCH_COMMON_WORDS:
            return True
    return False


def isEnglish(statement):
    words = statement.split()
    for word in words:
        if word.lower() in ENGLISH_COMMON_WORDS:
            return True
    return False


def is_dutch_dipthong(statement):
    words = statement.split()
    for word in words:
        if word.lower() in DUTCH_DIPTHONGS:
            return True
    return False


def is_english_article(statement):
    words = statement.split()
    for word in words:
        if word.lower() in ENGLISH_ARTICLES:
            return True
    return False


def avg_word_len(statement):
    words = statement.split()
    avg = sum(len(word) for word in words) / len(words)
    return avg > 5.0


def non_english(statement):
    words = statement.split()
    for word in words:
        for token in word:
            if not re.match("^[A-Za-z0-9_-]*$", token):
                return True
    return False


def get_features(data, train):
    feature_matrix = []
    for row in data:
        feature = []
        if train:
            statement = row.split('|')[-1]
        else:
            statement = row
        feature.append(avg_word_len(statement))
        feature.append(is_english_article(statement))
        feature.append(is_dutch_dipthong(statement))
        feature.append(isDutch(statement))
        feature.append(isEnglish(statement))
        feature.append(non_english(statement))
        if train:
            feature.append(row.split('|')[0])
        feature_matrix.append(feature)
    return feature_matrix
