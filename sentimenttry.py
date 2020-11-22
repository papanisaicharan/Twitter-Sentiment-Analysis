
"""
Created on Sat Dec  8 15:05:18 2018

@author: saicharan
"""
import sys,os,re,try1
import time,random,nltk
import stanfordcorpus
import preprocessing
from functools import wraps
import matplotlib.pyplot as plt

def draw_result(x, y, x1,y1, title,filename):
    plt.plot(x, y, '-r')

    plt.xlabel(x1)
    plt.ylabel(y1)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig("results/"+filename+".png")  # should before plt.show method
    #plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')
    #plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)
    #plt.scatter(x,y, label='skitscat', color='k', s=25, marker="o")
    
    plt.show()
def draw_result1(x, y, x1,y1, title,filename):
    #plt.plot(x, y, '-r')

    plt.xlabel(x1)
    plt.ylabel(y1)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.bar(x,y, label="Accuracies", color='b')
    plt.savefig("results/"+filename+".png")  # should before plt.show method
    
    #plt.hist(x, y, histtype='bar', rwidth=0.8)
    #plt.scatter(x,y, label='skitscat', color='k', s=25, marker="o")
    
    plt.show()

def get_time_stamp():
    return time.strftime("%y%m%d-%H%M%S-%Z")

# refer this for yield https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/
"""At a glance, the yield statement is used to define generators,
 replacing the return of a function to provide a result to its caller 
 without destroying local variables. Unlike a function, 
 where on each call it starts with new set of variables,
 a generator will resume the execution where it was left off"""
def grid(alist, blist):
    for a in alist:
        for b in blist:
            yield(a, b)

TIME_STAMP = get_time_stamp()

NUM_SHOW_FEATURES = 100
SPLIT_RATIO = 0.9
FOLDS = 10

"""classifiers"""
LIST_CLASSIFIERS = ['NaiveBayesClassifier','MaxentClassifier','DecisionTreeClassifier','SvmClassifier']
LIST_METHODS = ['1step', '2step']


##############################################################################################################

def getTrainingAndTestData(tweets, K, k, method, feature_set):
    
    add_ngram_feat = feature_set.get('ngram', 1)
    add_negtn_feat = feature_set.get('negtn', False)
    
    procTweets = [ (preprocessing.processAll(text, subject=subj, query=quer), sent)    \
                        for (text, sent, subj, quer) in tweets]
        #refer this http://www.nltk.org/howto/stem.html
    stemmer = nltk.stem.PorterStemmer()

    all_tweets = []                                             #DATADICT: all_tweets =   [ (words, sentiment), ... ]
    for (text, sentiment) in procTweets:
        words = [word if(word[0:2]=='__') else word.lower() for word in text.split() if len(word) >= 3]
        words = [stemmer.stem(w) for w in words]                #DATADICT: words = [ 'word1', 'word2', ... ]
        all_tweets.append((words, sentiment))
        
    train_tweets = [x for i,x in enumerate(all_tweets) if i % K !=k]
    test_tweets  = [x for i,x in enumerate(all_tweets) if i % K ==k]
    
    unigrams_fd = nltk.FreqDist()
    if add_ngram_feat > 1 :
        n_grams_fd = nltk.FreqDist()
    
    for( words, sentiment ) in train_tweets:
        words_uni = words
        unigrams_fd.update(words)

        if add_ngram_feat>=2 :
            words_bi  = [ ','.join(map(str,bg)) for bg in nltk.bigrams(words) ]
            n_grams_fd.update( words_bi )

        if add_ngram_feat>=3 :
            words_tri  = [ ','.join(map(str,tg)) for tg in nltk.trigrams(words) ]
            n_grams_fd.update( words_tri )

    sys.stderr.write( '\nlen( unigrams ) = '+str(len( unigrams_fd.keys() )) )
    
    #unigrams_sorted = nltk.FreqDist(unigrams).keys()
    p=[]
    q=[]
    for i,x in unigrams_fd.most_common(200):
        p.append(i)
        q.append(x)
    draw_result(p,q, "Words","frequency", "Unigrams","Unigrams_"+str(k))
    #bigrams_sorted = nltk.FreqDist(bigrams).keys()
    #trigrams_sorted = nltk.FreqDist(trigrams).keys()

    if add_ngram_feat > 1 :
        sys.stderr.write( '\nlen( n_grams ) = '+str(len( n_grams_fd )) )
        #p.setText('\nlen( n_grams ) = '+str(len( n_grams_fd ))+'\n') 
        ngrams_sorted = [ k for (k,v) in n_grams_fd.items() if v>1]
        p1=[]
        q1=[]
        for i,x in n_grams_fd.most_common(200):
            p1.append(i)
            q1.append(x)
        sys.stderr.write( '\nlen( ngrams_sorted ) = '+str(len( ngrams_sorted )) )
        draw_result(p1,q1, "Words","frequency", "bigrams","bigrams_"+str(k))
        #p.setText( '\nlen( ngrams_sorted ) = '+str(len( ngrams_sorted ))+'\n') 
    
    ####################################################################################
    def get_word_features(words):
        bag = {}
        words_uni = [ 'has(%s)'% ug for ug in words ]

        if( add_ngram_feat>=2 ):
            words_bi  = [ 'has(%s)'% ','.join(map(str,bg)) for bg in nltk.bigrams(words) ]
        else:
            words_bi  = []

        if( add_ngram_feat>=3 ):
            words_tri = [ 'has(%s)'% ','.join(map(str,tg)) for tg in nltk.trigrams(words) ]
        else:
            words_tri = []

        for f in words_uni+words_bi+words_tri:
            bag[f] = 1

        #bag = collections.Counter(words_uni+words_bi+words_tri)
        return bag
    
    #https://docs.python.org/3/library/re.html#re.X
    negtn_regex = re.compile( r"""(?:
        ^(?:never|no|nothing|nowhere|noone|none|not|
            havent|hasnt|hadnt|cant|couldnt|shouldnt|
            wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
        )$
    )
    |
    n't
    """, re.X)
    
    def get_negation_features(words):
        INF = 0.0
        negtn = [ bool(negtn_regex.search(w)) for w in words ]

        left = [0.0] * len(words)
        prev = 0.0
        for i in range(0,len(words)):
            if( negtn[i] ):
                prev = 1.0
            left[i] = prev
            prev = max( 0.0, prev-0.1)

        right = [0.0] * len(words)
        prev = 0.0
        for i in reversed(range(0,len(words))):
            if( negtn[i] ):
                prev = 1.0
            right[i] = prev
            prev = max( 0.0, prev-0.1)

        return dict( zip(
                        ['neg_l('+w+')' for w in  words] + ['neg_r('+w+')' for w in  words],
                        left + right ) )
		#{[neg_l("saicharan"),neg_l("pavan")]:[1.0,0.0]}
    
    #http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
    def counter(func):  
        @wraps(func)
        def tmp(*args, **kwargs):
            tmp.count += 1
            return func(*args, **kwargs)
        tmp.count = 0
        return tmp
    #http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
    @counter    
    def extract_features(words):
        features = {}
        #this will also removes the duplicates
        word_features = get_word_features(words)
        features.update( word_features )
		#duplicates are removed
        if add_negtn_feat :
            negation_features = get_negation_features(words)
            features.update( negation_features )
            
        sys.stderr.write( '\rfeatures extracted for ' + str(extract_features.count) + ' tweets')
        #p.setText( '\rfeatures extracted for ' + str(extract_features.count) + ' tweets'+'\n')
        return features

    extract_features.count = 0;
    ####################################################################################
    if( '1step' == method ):
        # Apply NLTK's Lazy Map
        v_train = nltk.classify.apply_features(extract_features,train_tweets)
        v_test  = nltk.classify.apply_features(extract_features,test_tweets)
        return (v_train, v_test)

    elif( '2step' == method ):
        isObj   = lambda sent: sent in ['neg','pos']
        makeObj = lambda sent: 'obj' if isObj(sent) else sent

        train_tweets_obj = [ (words, makeObj(sent)) for (words, sent) in train_tweets ]
        test_tweets_obj  = [ (words, makeObj(sent)) for (words, sent) in test_tweets ]

        train_tweets_sen = [ (words, sent) for (words, sent) in train_tweets if isObj(sent) ]
        test_tweets_sen  = [ (words, sent) for (words, sent) in test_tweets if isObj(sent) ]

        v_train_obj = nltk.classify.apply_features(extract_features,train_tweets_obj)
        v_train_sen = nltk.classify.apply_features(extract_features,train_tweets_sen)
        v_test_obj  = nltk.classify.apply_features(extract_features,test_tweets_obj)
        v_test_sen  = nltk.classify.apply_features(extract_features,test_tweets_sen)

        test_truth = [ sent for (words, sent) in test_tweets ]

        return (v_train_obj,v_train_sen,v_test_obj,v_test_sen,test_truth)

    else:
        return nltk.classify.apply_features(extract_features,all_tweets)
    




#################################################################################

def trainAndClassify( tweets, classifier, method, feature_set, fileprefix ):
    INFO = '_'.join( [str(classifier), str(method)] + [ str(k)+str(v) for (k,v) in feature_set.items()] )
    # NaiveBayesClassifier_1step_ngram1_negtnFalse
    if( len(fileprefix)>0 and '_'!=fileprefix[0] ):
        directory = os.path.dirname(fileprefix)
        #print(directory)
        #it's output logs
        if not os.path.exists(directory):
            os.makedirs(directory)
        realstdout = sys.stdout
        #now realstdout can be used for printing on console and sys.stdout is assigned filepointer
        sys.stdout = open( fileprefix+'_'+INFO+'.txt' , 'w')
        sys.stdout.write( INFO )
    
    sys.stderr.write( '\n'+ '#'*80 +'\n' + INFO )
                    
                     
    if('NaiveBayesClassifier' == classifier):
        #visit this for easy understanding https://pythonprogramming.net/naive-bayes-classifier-nltk-tutorial/
        CLASSIFIER = nltk.classify.NaiveBayesClassifier
        
        def train_function(v_train):
            return CLASSIFIER.train(v_train)
        
    elif('MaxentClassifier' == classifier):
        
        CLASSIFIER = nltk.classify.MaxentClassifier
        
        def train_function(v_train):
            return CLASSIFIER.train(v_train, algorithm='GIS', max_iter=10)
        
    elif('SvmClassifier' == classifier):
        
        CLASSIFIER = nltk.classify.SvmClassifier
        def SvmClassifier_show_most_informative_features( self, n=10 ):
            print('not implemented')
        CLASSIFIER.show_most_informative_features = SvmClassifier_show_most_informative_features
        def train_function(v_train):
            return CLASSIFIER.train(v_train)
        
    elif('DecisionTreeClassifier' == classifier):
        
        CLASSIFIER = nltk.classify.DecisionTreeClassifier
        
        def DecisionTreeClassifier_show_most_informative_features( self, n=10 ):
            text = ''
            for i in range( 1, 10 ):
                text = nltk.classify.DecisionTreeClassifier.pp(self,depth=i)
                if len( text.split('\n') ) > n:
                    break
            sys.stderr.write(text)
            print(text)
        CLASSIFIER.show_most_informative_features = DecisionTreeClassifier_show_most_informative_features
        def train_function(v_train):
            return CLASSIFIER.train(v_train, entropy_cutoff=0.05, depth_cutoff=100, support_cutoff=10, binary=False)
    
    accuracies = []
    
    if '1step' == method:
     for k in range(FOLDS):
         #creating 1 part as  testing data from 10 part of whole data
        
        (v_train, v_test) = getTrainingAndTestData(tweets, FOLDS, k, method, feature_set)
        sys.stderr.write( '\n[training start]' )
        #p.setText('\n[training start]\n')
        classifier_tot = train_function(v_train)
        sys.stderr.write( ' [training complete]' )
        
        print('######################')
        print('1 Step Classifier :', classifier)
        accuracy_tot = nltk.classify.accuracy(classifier_tot, v_test)
        print('Accuracy :', accuracy_tot)
        print('######################')
        print(classifier_tot.show_most_informative_features(NUM_SHOW_FEATURES))
        print('######################')
              # build confusion matrix over test set
        """A confusion matrix is a table that is often used to describe the performance 
        of a classification model (or "classifier") on a set of test data for 
        which the true values are known. """
        test_truth   = [s for (t,s) in v_test]
        test_predict = [classifier_tot.classify(t) for (t,s) in v_test]

        print('Accuracy :', accuracy_tot)
        print('Confusion Matrix')
        print(nltk.ConfusionMatrix( test_truth, test_predict ))

        accuracies.append( accuracy_tot )
     print("Accuracies:", accuracies)
     for i in range(10):
         accuracies[i] = accuracies[i]*100
         
     draw_result1([1,2,3,4,5,6,7,8,9,10],accuracies, "10-folds","accuaracies", "accuracy graph","accuracy_"+str(1))
     print("Average Accuracy:", str(sum(accuracies)/10))
     
    elif '2step' == method:
        # (v_train, v_test) = getTrainingAndTestData(tweets,SPLIT_RATIO, '1step', feature_set)

        # isObj   = lambda sent: sent in ['neg','pos']
        # makeObj = lambda sent: 'obj' if isObj(sent) else sent

        # def makeObj_tweets(v_tweets):
        #     for (words, sent) in v_tweets:
        #         print sent, makeObj(sent)
        #         yield (words, makeObj(sent))
        # def getSen_tweets(v_tweets):
        #     for (words, sent) in v_tweets:
        #         print sent, isObj(sent)
        #         if isObj(sent):
        #             yield (words, sent)

        
        # v_train_obj = makeObj_tweets( v_train )
        # v_test_obj = makeObj_tweets( v_test )

        # v_train_sen = getSen_tweets( v_train )
        # v_test_sen = getSen_tweets( v_test )
	
     accuracies = []
     for k in range(FOLDS):
        (v_train_obj, v_train_sen, v_test_obj, v_test_sen, test_truth) = getTrainingAndTestData(tweets, FOLDS, k, method, feature_set)

        sys.stderr.write( '\n[training start]' )
        #p.setText('\n[training start]\n')
        classifier_obj = train_function(v_train_obj)
        sys.stderr.write( ' [training complete]' )
        #p.setText(' [training complete]\n')

        sys.stderr.write( '\n[training start]' )
        #p.setText('\n[training start]\n')
        classifier_sen = train_function(v_train_sen)
        sys.stderr.write( ' [training complete]' )
        #p.setText('\n[training complete]\n')

        print('######################')
        print('Objectivity Classifier :', classifier)
        accuracy_obj = nltk.classify.accuracy(classifier_obj, v_test_obj)
        print('Accuracy :', accuracy_obj)
        print('######################')
        print(classifier_obj.show_most_informative_features(NUM_SHOW_FEATURES))
        print('######################')

        test_truth_obj   = [s for (t,s) in v_test_obj]
        test_predict_obj = [classifier_obj.classify(t) for (t,s) in v_test_obj]

        print('Accuracy :', accuracy_obj)
        print('Confusion Matrix')
        print(nltk.ConfusionMatrix( test_truth_obj, test_predict_obj ))

        print('######################')
        print('Sentiment Classifier :', classifier)
        accuracy_sen = nltk.classify.accuracy(classifier_sen, v_test_sen)
        print('Accuracy :', accuracy_sen)
        print('######################')
        print(classifier_sen.show_most_informative_features(NUM_SHOW_FEATURES))
        print('######################')

        test_truth_sen   = [s for (t,s) in v_test_sen]
        test_predict_sen = [classifier_sen.classify(t) for (t,s) in v_test_sen]

        print('Accuracy :', accuracy_sen)
        print('Confusion Matrix')
        if( len(test_truth_sen) > 0 ):
            print(nltk.ConfusionMatrix( test_truth_sen, test_predict_sen ))

        v_test_sen2 = [(t,classifier_obj.classify(t)) for (t,s) in v_test_obj]
        test_predict = [classifier_sen.classify(t) if s=='obj' else s for (t,s) in v_test_sen2]

        correct = [ t==p for (t,p) in zip(test_truth, test_predict)]
        accuracy_tot = float(sum(correct))/len(correct) if correct else 0

        print('######################')
        print('2 - Step Classifier :', classifier)
        print('Accuracy :', accuracy_tot)
        print('Confusion Matrix')
        print(nltk.ConfusionMatrix( test_truth, test_predict ))
        print('######################')

        classifier_tot = (classifier_obj, classifier_sen)
        accuracies.append( accuracy_tot )
     print("Accuracies:", accuracies)
     for i in range(10):
         accuracies[i] = accuracies[i]*100
         
     draw_result1([1,2,3,4,5,6,7,8,9,10],accuracies, "10-folds","accuaracies", "accuracy graph","accuracy_"+str(1))
     print("Average Accuracy:", str(sum(accuracies)/10))

    sys.stderr.write('\nAccuracies :')
    #p.setText('\nAccuracies :')
    for k in range(FOLDS):
        sys.stderr.write(' %0.5f'%accuracies[k])
    sys.stderr.write('\nAverage Accuracy: %0.5f\n'% (sum(accuracies)/FOLDS))
    #p.setText('\nAverage Accuracy: %0.5f\n'% (sum(accuracies)/FOLDS))
    sys.stderr.flush()

    sys.stdout.flush()
    if( len(fileprefix)>0 and '_'!=fileprefix[0] ):
        sys.stdout.close()
        sys.stdout = realstdout

    return classifier_tot

        
        





############################################################################################
############################################################################################
############################################################################################



def main(argv) :
    
    #####################################################################
    """python sentimenttry.py logs/fileprefix NaiveBayesClassifier,MaxentClassifier,DecisionTreeClassifier,SvmClassifier 1step,2step 1,3 0"""
    __usage__='''
    usage: python sentiment.py logs/fileprefix ClassifierName,s methodName,s ngramVal,s negtnVal,s
        example: python sentimenttry.py logs/fileprefix NaiveBayesClassifier,MaxentClassifier,DecisionTreeClassifier,SvmClassifier 1step,2step 1,3 0
        ClassifierName,s:   %s
        methodName,s:       %s
        ngramVal,s:         %s
        negtnVal,s:         %s
    ''' % ( str( LIST_CLASSIFIERS ), str( LIST_METHODS ), str([1,3]), str([0,1]) )
    
    
    fileprefix = ''
    if (len(argv) >= 1) :
        fileprefix = str(argv[0])
    else :
        fileprefix = 'logs/run'
    
    
    classifierNames = []
    if (len(argv) >= 2) :
        classifierNames = [ name for name in argv[1].split(',') if name in LIST_CLASSIFIERS ]
    else :
        classifierNames = ['NaiveBayesClassifier']


    methodNames = []    
    if (len(argv) >= 3) :
        methodNames = [name for name in argv[2].split(',') if name in LIST_METHODS]
    else :
        methodNames = ['1step']
    
    
    ngramVals = []
    if (len(argv) >= 4) :
        ngramVals = [int(val) for val in argv[3].split(',') if val.isdigit()]
        """ngramVals = []
        for val in argv[3].split(','):
            if val.isdigit():
                ngramVals.append(int(val))"""
    else :
        ngramVals = [ 2 ]


    negtnVals = []
    if (len(argv) >= 5) :
        negtnVals = [bool(int(val)) for val in argv[4].split(',') if val.isdigit()]
    else :
        negtnVals = [ False ]
        
        
    print(classifierNames, methodNames, ngramVals, negtnVals)
    if (len( fileprefix )==0 or len( classifierNames )==0 or len( methodNames )==0 or len( ngramVals )==0 or len( negtnVals )==0 ):
        print(__usage__)
        return
    
    tweets2 = stanfordcorpus.getNormalisedTweets('./stanfordcorpus/'+stanfordcorpus.FULLDATA+'.100000.norm.csv')

    random.shuffle(tweets2)
    
    sys.stderr.write("starting sentimental analysis")

    sys.stderr.write( '\nlen( tweets ) = '+str(len( tweets2 )) )

    sys.stderr.write( '\n' )
    sys.stdout.flush()
    
	
    try1.preprocessingStats( tweets2, fileprefix='logs/stats_'+TIME_STAMP+'/STAN')#logs/stats_'+TIME_STAMP+'/STAN' )

    #print(tweets2)
    
    #tweets2 = ["gave my mother her mother's day present. she loved it ", 'pos', 'NO_QUERY', []]
    """for ((k,x),y) in grid(grid( classifierNames, methodNames),ngramVals):
        print((k,x),y)
    output for above ('NaiveBayesClassifier', '1step') 1
    ('NaiveBayesClassifier', '1step') 3
    ('NaiveBayesClassifier', '2step') 1
    ('NaiveBayesClassifier', '2step') 3
    ('MaxentClassifier', '1step') 1
    ('MaxentClassifier', '1step') 3
    ('MaxentClassifier', '2step') 1
    ('MaxentClassifier', '2step') 3
    ('DecisionTreeClassifier', '1step') 1
    ('DecisionTreeClassifier', '1step') 3
    ('DecisionTreeClassifier', '2step') 1
    ('DecisionTreeClassifier', '2step') 3
    ('SvmClassifier', '1step') 1
    ('SvmClassifier', '1step') 3
    ('SvmClassifier', '2step') 1
    ('SvmClassifier', '2step') 3 
    for (((k,x),y),p) in grid(grid(grid( classifierNames, methodNames),ngramVals),negtnVals):
        print(((k,x),y),p)
    (('NaiveBayesClassifier', '1step'), 1) False
    (('NaiveBayesClassifier', '1step'), 3) False
    (('NaiveBayesClassifier', '2step'), 1) False
    (('NaiveBayesClassifier', '2step'), 3) False
    (('MaxentClassifier', '1step'), 1) False
    (('MaxentClassifier', '1step'), 3) False
    (('MaxentClassifier', '2step'), 1) False
    (('MaxentClassifier', '2step'), 3) False
    (('DecisionTreeClassifier', '1step'), 1) False
    (('DecisionTreeClassifier', '1step'), 3) False
    (('DecisionTreeClassifier', '2step'), 1) False
    (('DecisionTreeClassifier', '2step'), 3) False
    (('SvmClassifier', '1step'), 1) False
    (('SvmClassifier', '1step'), 3) False
    (('SvmClassifier', '2step'), 1) False
    (('SvmClassifier', '2step'), 3) False"""
    TIME_STAMP1 = get_time_stamp()
    #defined above
    for (((cname, mname), ngramVal), negtnVal) in grid( grid( grid( classifierNames, methodNames), ngramVals ), negtnVals ):
        try:
            #print(classifierNames, methodNames, ngramVals, negtnVals) 
            print("Attempting trainAndClassify with These parameters",cname,mname,ngramVal,negtnVal)
            trainAndClassify(
                tweets2, classifier=cname, method=mname,
                feature_set={'ngram':ngramVal, 'negtn':negtnVal},
                fileprefix=fileprefix+'_'+TIME_STAMP1 )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main(sys.argv[1:])
