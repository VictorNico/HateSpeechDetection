# importation de librairie

# module de manipulation des donnees
import re # regex to detect username, url, html entity
import nltk # to use word tokenize (split the sentence into words)
from nltk.corpus import stopwords # to remove the stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras import backend as K
import joblib
#notes : all of the function taking 1 text at a time
stop_words = set(stopwords.words('english'))
# add rt to remove retweet in dataset (noise)
stop_words.add("rt")

# remove html entity:
def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text

# change the user tags
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "user", raw_text)

    return text

# remove urls
def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '', raw_text)

    return text

# remove unnecessary symbols
def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')

    return text

# remove stopwords
def remove_stopwords(raw_text):
    tokenize = nltk.word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = " ".join(text)

    return text

## this function in to clean all the dataset by utilizing all the function above
def preprocess(datas):
    clean = []
    # change the @xxx into "user"
    clean = [change_user(text) for text in datas]
    # remove emojis (specifically unicode emojis)
    clean = [remove_entity(text) for text in clean]
    # remove urls
    clean = [remove_url(text) for text in clean]
    # remove trailing stuff
    clean = [remove_noise_symbols(text) for text in clean]
    # remove stopwords
    clean = [remove_stopwords(text) for text in clean]

    return clean

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisions = precision(y_true, y_pred)
    recalls = recall(y_true, y_pred)
    return 2*((precisions*recalls)/(precisions+recalls+K.epsilon()))



def predictor(data):
    if isinstance(data, list):
        clean_messages = preprocess(data)
        print(clean_messages)
        tokenizer = joblib.load('./models/tokenizer.sav')
        X_test = tokenizer.texts_to_sequences(clean_messages)
        max_length = joblib.load('./models/max_length.sav')
        X_test = pad_sequences(X_test, maxlen=max_length)
        model = load_model("./models/model.h5")
        prediction = model.predict(X_test).tolist()
        prediction = [preds.index(max(preds)) for preds in prediction]
        return prediction
    return None


if __name__ == '__main__':
    # nltk.download('punkt')
    # nltk.download('stopwords')
    data = [
        "!!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out...",
        '" momma said no pussy cats inside my doghouse "',
        '"@Addicted2Guys: -SimplyAddictedToGuys http://t.co/1jL4hi8ZMF" woof woof hot scally lad',
        '"@AllAboutManFeet: http://t.co/3gzUpfuMev" woof woof and hot soles',
        '"@Allyhaaaaa: Lemmie eat a Oreo &amp; do these dishes." One oreo? Lol',
        '"@ArizonasFinest6: Why the eggplant emoji doe?"y he say she looked like scream lmao',
        '"@BabyAnimalPics: baby monkey bathtime http://t.co/7KPWAdLF0R"\nAwwwwe! This is soooo ADORABLE!',
        '"@DomWorldPeace: Baseball season for the win. #Yankees" This is where the love started',
        '"@DunderbaIl: I\'m an early bird and I\'m a night owl, so I\'m wise and have worms."',
        '"@EdgarPixar: Overdosing on heavy drugs doesn\'t sound bad tonight." I do that pussy shit every day.'
    ]
    print(predictor(data))