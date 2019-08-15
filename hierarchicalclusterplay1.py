
import sys
import pickle
import nltk;
import numpy as np
import statistics;
import pandas as pd;
import re;
from numpy.f2py.auxfuncs import throw_error
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math;

import pickle
import datetime

def saveObject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# Customised stopword list

stopwords = nltk.corpus.stopwords.words('english')

stopwords.append('janurary');
stopwords.append('feburary');
stopwords.append('march');
stopwords.append('april');
stopwords.append('may');
stopwords.append('june');
stopwords.append('july');
stopwords.append('august');
stopwords.append('september');
stopwords.append('october');
stopwords.append('november');
stopwords.append('december');
stopwords.append('pleas');
stopwords.append('thank');
stopwords.append('ms');
stopwords.append('org');
stopwords.append('use');
stopwords.append('see');
stopwords.append('new');
stopwords.append('need');
stopwords.append('issue');
stopwords.append('issu');
stopwords.append('detail');
stopwords.append('version');
stopwords.append('test');
stopwords.append('would');
stopwords.append('could');
stopwords.append('should');
stopwords.append('fix');
stopwords.append('add');
stopwords.append('try');
stopwords.append('trying');
stopwords.append('patch');
stopwords.append('like');
stopwords.append('doesnt');
stopwords.append('dont');
stopwords.append('also');
stopwords.append('im');
stopwords.append('check');
stopwords.append('got');
stopwords.append('look');
stopwords.append('get');
stopwords.append('code');
stopwords.append('commit');
stopwords.append('make');
stopwords.append('think');
stopwords.append('run');
stopwords.append('remove');
stopwords.append('add');
stopwords.append('throw');
stopwords.append('throws');
stopwords.append('may');
stopwords.append('has');
stopwords.append('set');
stopwords.append('update');
stopwords.append('updat');
stopwords.append('updates');
stopwords.append('updated');
stopwords.append('looking');
stopwords.append('updating'); 
stopwords.append('ill');
stopwords.append('call');
stopwords.append('work');
stopwords.append('worked');
stopwords.append('working');
stopwords.append('works');
stopwords.append('br');
stopwords.append('copyright');
stopwords.append('article');
stopwords.append('title');
stopwords.append('document');
stopwords.append('group');
stopwords.append('control');
stopwords.append('placebo');
stopwords.append('h');
stopwords.append('documents');
stopwords.append('doc');
stopwords.append('id');
stopwords.append('sec');
stopwords.append('g');
stopwords.append('kg');
stopwords.append('mg');
stopwords.append('locate');
stopwords.append('cite');
stopwords.append('p');
stopwords.append('ci');
stopwords.append('study');
stopwords.append('vs');


# Set up our stemmer, which remembers roughly how we stemmed so we can do a reverse conversion if required

class Stemmer:
    
    
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        self.unstemDict= dict();
    
    
    def unStem(self, t):
        
        
        
        if t in self.unstemDict:
            return self.unstemDict[t];
        
        out = "";
        for x in t.split(" "):
            if x in self.unstemDict:
                out = out +self.unstemDict[x]+" " ;
            else:
                out = out + x +" ";
        
        return out;
    
    #Have some custom stemming rules over and above the default snowball stemmer
    def tweakedStem(self,t):
        t = re.sub(r'tter$','t', t)    
        t = re.sub(r'^email$','mail', t)
        t = re.sub(r'^e-mail$','mail', t)
        t = re.sub(r'^Email$','mail', t)
        
        s = self.stemmer.stem(t)
        s = re.sub(r'^repositori$','repo', s)
        s = re.sub(r'^github$','git', s)
        
        if not s in self.unstemDict or len(self.unstemDict[s])>len(t):
            self.unstemDict[s] = t;
        return s;
    
    def tokenize_and_stem(self,text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [self.tweakedStem(t) for t in filtered_tokens]
           
        
        return stems
    
    
    def tokenize_only(self,text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens


class NavHierarchyTopic:
    def __init__(self):
        self.documentsFilter = list();
        self.children = list();
        self.lda =None;
        self.depth = 0;
        self.parent = None;
        self.navTopicTuple = None;
        self.documentsByIndex = dict();
        self.documentsRankByIndex = dict();
        
        

class Ticket:
    def __init__(self,identifier, 
                title,
                body,
                journal):
        self.identifier = identifier;
        self.title = title;
        self.body = body;
        self.journal = journal;





class FixKMeansHierarchy:


    def clean(self, b) :
        b = re.sub( '[^a-z 0-9]',' ', b.lower());
        b = re.sub( '\\b[0-9]+\\b',' ', b);
        b = re.sub( ' +',' ', b);
        return b;
    
    def filterCorpus(self,corpus, documentsFilter):
        out = [];
        i = 0;
        for doc in corpus:
            b=documentsFilter[i];
            if b:
                out.append(doc);
            
            i=i+1;
        return out;
        
    def performance(self,filteredCorpus,kmeans, numTopics):
        
        return -kmeans.inertia_*math.sqrt(numTopics);
    
    def doFit(self, df):        
        MAXLEVELSPLIT = 8;
        MINLEVELSPLIT = 3;
        DESIRED_CLUSTER_SIZE = 20;
        DESIRED_COHESION = 0.7;
        MIN_SIZE = 5;
#        MAX_DIM = 10000;
        MAX_DIM = 200;
        


        tickets = [];
        for index, row in df.iterrows():
            tickets.append( Ticket(
                identifier= index, 
                title = row['title'],
                body = self.clean(row['body']),
                journal =  row['journal']
            ));
        print(tickets[0].body)

        
        stemmer = Stemmer();
        
        
        
        
        fullKMeansTopic=NavHierarchyTopic();
        fullKMeansTopic.rank = 0;
        fullKMeansTopic.kmeans = None;
        for d in tickets:
            fullKMeansTopic.documentsFilter.append(True) ;
            
        queue = [fullKMeansTopic];
        
        
        
        output=[fullKMeansTopic]
        
        while len(queue) > 0:
        
            bestPerformance=-1e99;
            bestKmeans = None;
            
            currentTopic = queue.pop();
            
            if (currentTopic.rank<DESIRED_COHESION):
                print( "-----------------");
                print( "Processing new level in K Means tree. Depth " + str(currentTopic.depth));
                                
                filteredTickets = self.filterCorpus(tickets, currentTopic.documentsFilter);        
        
                
                print( "Corpus size " + str(len(filteredTickets)));
                
                if (len(filteredTickets)<MIN_SIZE):
                    print( "Node too small for further classification");
                else :
                                
        
                    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=MAX_DIM,
                                         min_df=0.03, stop_words=stopwords,
                                         use_idf=True, tokenizer=stemmer.tokenize_and_stem, ngram_range=(1,4))
        
                    bodies = list();
                    for t in filteredTickets:
                        b = t.title.strip().lower() +' ' + t.body.strip().lower();
                        if (len(b)==0):
                            bodies.append("empty");
                        else:
                            bodies.append(b);
                    print( "Vectorise");

                    filteredCorpus = tfidf_vectorizer.fit_transform(bodies) #fit the vectorizer to bodies
                    
                    nj = -1;
                    if (len(filteredTickets)<100):
                        nj = 1;  
                    
                    
                    
                    numClusters = int(round(len(filteredTickets) / DESIRED_CLUSTER_SIZE));
                    if numClusters<MINLEVELSPLIT:
                        numClusters = MINLEVELSPLIT;
                    if numClusters>MAXLEVELSPLIT:
                        numClusters = MAXLEVELSPLIT;
                        
                    # Was a loop here
                        
                    print( "Exec KMeans clusters = " + str(numClusters));
                    kmeans = KMeans(n_clusters=numClusters, random_state=0, n_jobs=nj)
                    kmeans.fit_predict(filteredCorpus)
                    
                    p = self.performance(filteredCorpus,kmeans,numClusters);
                    print( "Splitting at " + str(numClusters) +" Topics yields perfomance " );
                    print( p );
                    if (p > bestPerformance):
                        bestKmeans = (kmeans, numClusters);
                        bestPerformance = p; 
                        print( "This is the best split at this level so far");
                    
                    # loop end
                        
                    
                    kmeans,numClusters = bestKmeans;
                    for clusterLabel in range(0,numClusters):
                        print( "Label : ");
                        print( clusterLabel);
                        
                        newTopic = NavHierarchyTopic()
                        newTopic.kmeans = kmeans;
                        newTopic.featureNames = [ stemmer.unStem(x) for x in tfidf_vectorizer.get_feature_names() ];
                        clusterCentroid = kmeans.cluster_centers_[clusterLabel].reshape(1,-1);
                        wordWeights=list();
                        for z in range(0, len(clusterCentroid[0])):
                            if clusterCentroid[0][z]!=0:
                                wordWeights.append((stemmer.unStem(newTopic.featureNames[z]), clusterCentroid[0][z]));
                        
                        wordWeights.sort(key=lambda x: x[1], reverse=True)
                        del wordWeights[min(8, len(wordWeights)) : len(wordWeights)];
                        print(wordWeights);

                        newTopic.navTopicTuple = (kmeans, wordWeights);                
                        newTopic.centroid = clusterCentroid[0];
                         
                        newTopic.parent = currentTopic;
                        newTopic.depth = currentTopic.depth+1;
                        z=0;
                        sumRank = 0;
                        for j in range(0,len(tickets)):
                            contained=False;
                            
                                                   
                            if currentTopic.documentsFilter[j]:
                                d=tickets[j];
                                thisDocClusterLabel = kmeans.labels_[z]
                                                     
                                if (clusterLabel==thisDocClusterLabel):
                                    rank = cosine_similarity(filteredCorpus[z],clusterCentroid)[0][0];
    #                                 print( "Cluster " + str(clusterLabel) + " contains " + d.title                           );
    #                                 print( ran);
                                    newTopic.documentsByIndex[j] = d;
                                    newTopic.documentsRankByIndex[j] = rank
                                    sumRank = sumRank + rank;
                                    contained = True
                                    
                                
                                z=z+1;   
                            
                            newTopic.documentsFilter.append(contained);
                         
                        if len(newTopic.documentsByIndex)>1:
                            newTopic.rank = sumRank / float(len(newTopic.documentsByIndex))
                            print( "Overall rank");
                            print( newTopic.rank);
                            currentTopic.children.append(newTopic);
                            queue.append(newTopic)
                        else:
                            print( "Ignoring trivial splits"                   );
    
                    
        
        print( "Hierarchical Clustering completed.");
            
        saveObject(output, './cluster_hierarchy.pickle');    
    
    
    
if __name__ == '__main__':
    df = pd.read_pickle('./ncbi.pickle');
    FixKMeansHierarchy().doFit(df);