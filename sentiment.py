# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 15:05:18 2018

@author: saicharan
"""
"""import preprocessing

print(preprocessing.countHandles("hello snjdn @nen @sai"))

print(dir(preprocessing))

text = "HairByJess,@iamjazzyfizzle I Wish I got to WATCH it with you!! I miss you and @iamlilnicki  how was the premiere?!"
words = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if ( (not False) | (len(word) >= 3) ) ]

print(words)"""

import preprocessing 
import csv,os,sys,collections
import nltk, pylab,numpy,math

def printFreqDistCSV( dist, filename='' ):
    n_samples = len(dist.keys())
    n_repeating_samples = sum([ 1 for (k,v) in dist.items
        () if v>1 ])
    n_outcomes = dist._N
    print('%-12s %-12s %-12s'% ( 'Samples', 'RepSamples', 'Outcomes' ))
    print( n_samples, n_repeating_samples, n_outcomes )
    
    if( len(filename)>0 and '_'!=filename[0] ):
        with open( filename, 'w+' ) as fcsv:
            distwriter = csv.writer( fcsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC )
            
            for (key,value) in dist.items():
                distwriter.writerow( [key, value] ) #print key, '\t,\t', dist[key]



def printFeaturesStats( tweets ):
    #no error in this,it will be printing in file
    arr_Handles   = numpy.array( [0]*len(tweets) )
    arr_Hashtags  = numpy.array( [0]*len(tweets) )
    arr_Urls      = numpy.array( [0]*len(tweets) )
    arr_Emoticons = numpy.array( [0]*len(tweets) )
    arr_Words     = numpy.array( [0]*len(tweets) )
    arr_Chars     = numpy.array( [0]*len(tweets) )
    

    i=0
    for (text, sent, subj, quer) in tweets:
        arr_Handles[i]   = preprocessing.countHandles(text)
        arr_Hashtags[i]  = preprocessing.countHashtags(text)
        arr_Urls[i]      = preprocessing.countUrls(text)
        arr_Emoticons[i] = preprocessing.countEmoticons(text)
        arr_Words[i]     = len(text.split())
        arr_Chars[i]     = len(text)
        i+=1

    print('%-10s %-010s %-4s '%('Features',  'Average',            'Maximum'))
    print('%10s %10.6f %10d'%('Handles',   arr_Handles.mean(),   arr_Handles.max()   ))
    print('%10s %10.6f %10d'%('Hashtags',  arr_Hashtags.mean(),  arr_Hashtags.max()  ))
    print('%10s %10.6f %10d'%('Urls',      arr_Urls.mean(),      arr_Urls.max()      ))
    print('%10s %10.6f %10d'%('Emoticons', arr_Emoticons.mean(), arr_Emoticons.max() ))
    print('%10s %10.6f %10d'%('Words',     arr_Words.mean(),     arr_Words.max()     ))
    print('%10s %10.6f %10d'%('Chars',     arr_Chars.mean(),     arr_Chars.max()     ))


def printReductionStats( tweets, function, filtering=True):
    #understood
    if( function ):
        procTweets = [ (function(text, subject=subj, query=quer), sent)    \
                        for (text, sent, subj, quer) in tweets]
    else:
        procTweets = [ (text, sent)    \
                        for (text, sent, subj, quer) in tweets]
    tweetsArr = []
    for (text, sentiment) in procTweets:
        words = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if ( (not filtering) | (len(word) >= 3) ) ]
        tweetsArr.append([words, sentiment])
    # tweetsArr
    #example
    #for word in ['red', 'blue', 'red', 'green', 'blue', 'blue']:
    #    cnt[word] += 1
    #cnt
    #Counter({'blue': 3, 'red': 2, 'green': 1})
    bag = collections.Counter()
    #https://docs.python.org/3.1/library/collections.html
    for (words, sentiment) in tweetsArr:
        bag.update(words)
    # unigram

    print('%20s %-10s %12d'% (
                ('None' if function is None else function.__name__),
                ( 'gte3' if filtering else 'all' ),
                sum(bag.values())
            ))
    return True



def printAllRecuctionStats(tweets):
    print('%-20s %-10s %-12s'% ( 'Preprocessing', 'Filter', 'Words' ))
    printReductionStats( tweets, None,                   False   )
    #printReductionStats( tweets, None,                   True    )
    printReductionStats( tweets, preprocessing.processHashtags,        True    )
    printReductionStats( tweets, preprocessing.processHandles,         True    )
    printReductionStats( tweets, preprocessing.processUrls,            True    )
    printReductionStats( tweets, preprocessing.processEmoticons,       True    )
    printReductionStats( tweets, preprocessing.processPunctuations,    True    )
    printReductionStats( tweets, preprocessing.processRepeatings,      True    )
    #printReductionStats( tweets, preprocessing.processAll,             False   )
    printReductionStats( tweets, preprocessing.processAll,             True    )



def preprocessingStats( tweets, fileprefix='' ):
    #print(tweets)
    #print("\n \n \n \n")
    
    if( len(fileprefix)>0 and '_'!=fileprefix[0] ):
        directory = os.path.dirname(fileprefix)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('writing to', fileprefix+'_stats.txt')
        #what ever i print it has to print in file
        realstdout = sys.stdout
        sys.stdout = open( fileprefix+'_stats.txt' , 'w') 
    
    
    ###########################################################################  

    print('for', len(tweets), 'tweets:')

    print('###########################################################################')

    printFeaturesStats( tweets )

    print('###########################################################################')

    printAllRecuctionStats( tweets )

    print('###########################################################################')

    
    procTweets = [ (preprocessing.processAll(text, subject=subj, query=quer), sent)    \
                        for (text, sent, subj, quer) in tweets]
    #print(procTweets)
    tweetsArr = []
    #it will eliminate all words whose length is less than 3 and lower the remaining
    for (text, sentiment) in procTweets:
        words = [word if(word[0:2]=='__') else word.lower() \
                        for word in text.split() \
                        if ( (len(word) >= 3) ) ]
        tweetsArr.append([words, sentiment])
    #print(tweetsArr)
    
    unigrams_fd = nltk.FreqDist()
    bigrams_fd = nltk.FreqDist()
    trigrams_fd = nltk.FreqDist()
    #print(nltk.bigrams(words))
    for (words, sentiment) in tweetsArr:
        words_bi = [ ','.join(map(str,bg)) for bg in nltk.bigrams(words) ]
        #print(words_bi)
        words_tri  = [ ','.join(map(str,tg)) for tg in nltk.trigrams(words) ]
        #print(words_tri)
        #refer this for update http://www.nltk.org/howto/probability.html 
        unigrams_fd.update( words )
        bigrams_fd.update( words_bi )
        trigrams_fd.update( words_tri )   
        
    print('Unigrams Distribution')
    printFreqDistCSV(unigrams_fd, filename=fileprefix+'_1grams.csv')
    if( len(fileprefix)>0 and '_'!=fileprefix[0] ):
        pylab.show = lambda : pylab.savefig(fileprefix+'_1grams.pdf')
    unigrams_fd.plot(50, cumulative=True)
    pylab.close()
    
    print('Bigrams Distribution')
    printFreqDistCSV(bigrams_fd, filename=fileprefix+'_2grams.csv')
    if( len(fileprefix)>0 and '_'!=fileprefix[0] ):
        pylab.show = lambda : pylab.savefig(fileprefix+'_2grams.pdf')
    bigrams_fd.plot(50, cumulative=True)
    pylab.close()

    print('Trigrams Distribution')
    printFreqDistCSV(trigrams_fd, filename=fileprefix+'_3grams.csv')
    if( len(fileprefix)>0 and '_'!=fileprefix[0] ):
        pylab.show = lambda : pylab.savefig(fileprefix+'_3grams.pdf')
    trigrams_fd.plot(50, cumulative=True)
    pylab.close() 

    if( len(fileprefix)>0 and '_'!=fileprefix[0] ):
        sys.stdout.close()
        sys.stdout = realstdout
        
