import nltk
import numpy as np
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


## download stopwords
nltk.download('stopwords')
## download punctuations
nltk.download('punkt')


SMS = pd.read_table('data/SMSSpamCollection', header=None)

# encode label
y_encoder = LabelEncoder()
y_data = y_encoder.fit_transform(SMS[0])


## cleaning
## replace email addresess with 'emailaddress'
process = SMS[1].str.replace(r'^\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$', 'emailaddress')
## replace urls with 'webaddresess'
process = process.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddresess')
## replace money symbols with 'moneysymb'
process = process.str.replace(r'£|\$', 'webaddresess')
## replace phone numbers with 'phonenumber'
process = process.str.replace(r'^[2-9]\d{2}-\d{3}-\d{4}$', 'phonenumber')
## replace normal numbers with 'numbers'
process = process.str.replace(r'\d+(\.\d+)?', 'numbers')

# remove punctuation
process = process.str.replace(r'[^\w\d\s]', ' ')
# replace whitespace between terms
process = process.str.replace(r'\s+', ' ')
# remove leading and trailing whitespace
process = process.str.replace(r'^\s+|\s+?$', '')

# change the words to lower case 
process = process.str.lower()


# remove stop words from text
stop_words = set(stopwords.words('english'))
process = process.apply(lambda x:' '.join(term for term in x.split() if term not in stop_words))


# remove word stems using a porter stemmer
ps = nltk.PorterStemmer()
process = process.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))



all_words = []
for msg in process:
    words = word_tokenize(msg)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)
word_fature = list(all_words.keys())[:1500]

# define a find_feature function
def find_feature(message):
    words = word_tokenize(message)
    features = {}
    for word in word_fature:
        features[word] = (word in words)
        
    return features


# zip message and the labels
message = list(zip(process, y_data))

seed = 1
np.random.seed = seed
np.random.shuffle(message)

# call find_feature func for all message
featureset = [(find_feature(text), label) for (text, label) in message]

# spliting training and testing data set
training , testing = train_test_split(featureset, test_size=0.2, random_state=seed)



##############
#### model####
##############
names = ['K Neighbors', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'SGD Classifier', 'SVM linear']
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))


# wrap models in nltk 
for name, model in models:
    nltk_model = SklearnClassifier(model)
    ## A list of `(featureset, label)` where each featureset is a `dict` mapping `strings` to either numbers, booleans or strings.
    nltk_model.train(training) 
    accuracy = nltk.classify.accuracy(nltk_model, testing)
    print(f'{name} accuracy is {accuracy}')

# ensemble method - voting classifier
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators=models, voting='hard',  n_jobs=-1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_ensemble, testing)
print(f'ensemble method accuracy is {accuracy}')

# make classlable prediction for testing set
txt_feature, labels = zip(*testing)
prediction = nltk_ensemble.classify_many(txt_feature)

# print classification report 
print(classification_report(labels, prediction))




