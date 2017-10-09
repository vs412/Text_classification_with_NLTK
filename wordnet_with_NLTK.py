from nltk.corpus import wordnet

syns = wordnet.synsets("party")

#print (syns[0].definition())

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')

print (w1.wup_similarity(w2))
