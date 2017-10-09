import nltk
import pickle
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

find_features(movie_reviews.words('neg/cv000_29416.txt'))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#Classifier = nltk.NaiveBayesClassifier.train(training_set)

Classifier_f = open("naivebayes.pickle", "rb")
Classifier = pickle.load(Classifier_f)
Classifier_f.close()

#print ("Classifier accuracy percent:", (nltk.classify.accuracy(Classifier, testing_set))*100)

#Classifier.show_most_informative_features(15)
'''
save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(Classifier, save_classifier)
save_classifier.close()

'''
