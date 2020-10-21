'''art.py -- Approximate Randomization Test

This script carries out a significance test on the output of an
instance-based machine learner based on the theory of
approximate randomization tests:

Eric W. Noreen, Computer-intensive Methods for Testing Hypotheses: An Introduction, John Wiley & Sons, New York, NY, USA, 1989.

No assumptions are made on the distribution of the variables. The only assumption made is that there are
no inter-instance dependencies, i.e. knowing the class label of 1 instance should not help
knowing the class label of another instance. This assumption is violated in the output from the MBT (memory-based tagger).

A nice example of why no inter-instance dependencies should be present is in:

Alexander Yeh, More accurate tests for the statistical significance of result differences,
 in: Proceedings of the 18th International Conference on Computational Linguistics, Volume 2,
 pages 947-953, 2000.


TEST STATISTICS

At the moment, the test statitics tested are differences in macro-recall, macro-precision, macro-f-score, micro-f-score, and accuracy.
This can be changed by changing the getscores() function.


DEPENDENCIES
    This script depends on confusionmatrix.py and combinations.py (www.clips.ua.ac.be/~vincent/software.html)
    and optionally scipy (www.scipy.org).

Copyright (c) 2013 CLiPS. All rights reserved.

# License: GNU General Public License, see http://www.clips.ua.ac.be/~vincent/scripts/LICENSE.txt


#Speed improvements by Sam Henry (July 2019)
'''
__author__="Vincent Van Asch"
__date__="September 2013"
__version__="3.0.3"
__url__ = 'http://www.clips.ua.ac.be/~vincent/software.html'

import sys, os, time
import random
import getopt
import datetime
from math import pow

from scipy.stats import binom_test
import confusionmatrix
import combinations


def loginfo(s): 
    print(('%s: %s' %(time.strftime('%d/%m/%Y %H:%M:%S'), s)))


def fread(fname, index=None, sep=None, encoding='utf8'):
    '''Reads in files as lists.
    
    sep: feature separator
    index: if None, the elements of the output list are the full lines
           if int,  the elements of the output list are string at position index
           if tuple, the elements of the output list slices from the full lines (as lists)  
    '''
    output=[]
    
    with open(os.path.abspath(os.path.expanduser(fname)), 'rU') as f:
        for l in f:
            line = l.strip()
            if line:
                if index is None:
                    output.append(line)
                else:
                    line = line.split(sep)
                    if isinstance(index, int):
                        output.append(line[index])
                    elif isinstance(index, (list, tuple)):
                        if len(index) != 2: raise ValueError('index should have length 2 not %d' %len(index))
                        output.append(line[index[0]:index[1]])
                    else:
                        raise TypeError('index should be None, int or tuple')
        
    
    return output

def strata_read(fname, sep=None, encoding='utf8'):
    out={}
    with open(os.path.abspath(os.path.expanduser(fname)), 'rU') as f:
        for l in f:
            line = l.strip()
            if line:
                parts = line.split(sep)
                stratum = parts[0]
                group = parts[1]
                data = [float(x) for x in parts[2:]]
                
                if stratum in list(out.keys()):
                    out[stratum][group] = data
                else:
                    out[stratum] = {group:data}
                    
    return out
                

MBTSEP = '\x13'
def mbtread(fname, sep="<utt>"):
    '''Reads in the sentences from an mbt format file.
    
    sep: sentence seperator (empty lines are also considered as sentence
    boundaries)
    
    Returns a list of strings. 
    Each string are the concatenated token labels from 1 sentence'''
    output=[]
    sentence=[]
    
    with open(os.path.abspath(os.path.expanduser(fname)), 'rU') as f:
        for l in f:
            line = l.strip()
            if line and line != sep:
                sentence.append(line.split()[-1])
            else:
                if sentence: output.append(MBTSEP.join(sentence))
                sentence=[]
                
    if sentence: output.append(MBTSEP.join(sentence))
                
    return output


def readtraining(fname, index=-1, sep=None):
    '''Reads in training and returns a dictionary with the distribution of the 
    classes in training'''
    d={}    
    for label in fread(fname, sep=sep, index=index):
        try:
            d[label]+=1
        except KeyError:
            d[label] = 1
    return d

def signtest(gold, system1, system2):
    '''Sign test for labeling accuracy'''
    assert len(gold) == len(system1) == len(system2)
    # Get all number where system1 is correct and the other false
    s1correct=0
    s2correct=0
    wrong=0
    for g, s1, s2 in zip(gold, system1, system2):
        if g==s1:
            s1correct+=1
        elif g==s2:
            s2correct+=1
        else:
            wrong+=1
        
    # The total number of predictions that are only correctly predicted
    # by 1 system
    total = s1correct+s2correct
    
    # make sure we test the smallest because of
    # bug with unequal N in binom_test
    correct = min([s1correct, s2correct])
    
    return binom_test(correct, total)

    
    
def termsigntest(gold, system1, system2):
    '''Sign test for term extraction recall'''
    print('WARNING: this function has not been validated')
    # True postives for only 1 system
    s1correct=0
    s2correct=0
    fn=0
    for t in gold:
        if t in system1:
            if t not in system2:
                s1correct+=1
        elif t in system2:
            s2correct+=1
        else:
            fn +=1
            
    # The total number of predictions that are only correctly predicted
    # by 1 system
    total = s1correct+s2correct
    
    return binom_test(s1correct, total)

    
def getscores(gold, system, training=None):
    '''
    Takes a gold and system list and returns a dictionary with
    macro-recall, macro-precision, macro-f-score, micro-f-score, accuracy.
    If training is given it uses the class label counts from training to compute the scores.
    
    gold: a list of class labels
    system: a list of class labels (in the same order as gold)
    training: a dictionary:
                key: class label
                value: number of occurrences
        
    Returns a dictionary:
        key: performance measure name
        value: performance score
    '''
    # Get confusion matrix
    assert len(gold) == len(system)
    
    # light mode for speed
    cm = confusionmatrix.ConfusionMatrix(light=True)

    # Add training
    if training:
        for k, v in list(training.items()):
            for i in range(v):
                cm.add_training([k])
    
    # Add data
    for g, s in zip(gold, system):
        cm.single_add(g, s)

    output={'macro-av. recall': cm.averaged(level=confusionmatrix.MACRO, score=confusionmatrix.RECALL, training=bool(training)), \
            'macro-av. precision': cm.averaged(level=confusionmatrix.MACRO, score=confusionmatrix.PRECISION, training=bool(training)), \
            'macro-av. f-score': cm.averaged(level=confusionmatrix.MACRO, score=confusionmatrix.FSCORE, training=bool(training)), \
            'micro-av. f-score': cm.averaged(level=confusionmatrix.MICRO, score=confusionmatrix.FSCORE, training=bool(training)), \
            'micro-av. precision': cm.averaged(level=confusionmatrix.MICRO, score=confusionmatrix.PRECISION, training=bool(training)), \
            'micro-av. recall': cm.averaged(level=confusionmatrix.MICRO, score=confusionmatrix.RECALL, training=bool(training)), \
            'lfb-micro-av. f-score': cm.averaged(level=confusionmatrix.MICROt, score=confusionmatrix.FSCORE, training=bool(training)), \
            'lfb-micro-av. precision': cm.averaged(level=confusionmatrix.MICROt, score=confusionmatrix.PRECISION, training=bool(training)), \
            'lfb-micro-av. recall': cm.averaged(level=confusionmatrix.MICROt, score=confusionmatrix.RECALL, training=bool(training)), \
            'accuracy': cm.accuracy()}

    return output



def getscores2(gold, system, training=None):    
    #print("Calculating P:" + str(datetime.datetime.now()))
    #overlap = float(len([i for i in system if i in gold]))
    overlap = float(len(system.intersection(gold)))
    P = overlap / len(system)
    R = overlap / len(gold)
    
    if P==0 or R==0:
        F=0.0
    else:
        F = 2*P*R/(P+R)    
    return {'recall': R, 'precision':P, 'f1-score': F}


def getscoresmbt(gold, system, training=None):
    '''Returns the mbt accuracy for the sentence'''
    correct=0
    total=0
    for g, s in zip(gold, system):
        g = g.split(MBTSEP)
        s = s.split(MBTSEP)
        
        assert len(g) == len(s)
        total += len(g)
        
        for gi, si in zip(g, s):
            if gi == si:
                correct+=1
                
    return {'accuracy': correct/float(total)}
    
        
def getscoresmbtmulti(gold, system, training=None, sep='_'):
    '''Returns scores for mbt'''
    # Create the yielder
    def reader(gold, system):
        for g,s in zip(gold, system):
            g = g.split(MBTSEP)
            s = s.split(MBTSEP)
            assert len(g) == len(s)
            
            for gi, si in zip(g, s):
                gi = set(gi.split(sep))
                si = set(si.split('_'))
                yield gi, si
    r = reader(gold, system)
            
    cm = confusionmatrix.ConfusionMatrix(compute_none=True)
    for g, p in r:
        cm.add(list(g), list(p))
        
    out={}
    for label in cm.labels:
        out[label] = cm.fscore(label)

    out['micro-fscore']=cm.averaged(level=confusionmatrix.MICRO, score=confusionmatrix.FSCORE)
    
    return out

    

def average(dumy, values, training=None):
    return {'mean': sum(values)/float(len(values))}
        
def teststatistic(gold, system1, system2, training=None, scoring=getscores, absolute=True):
    '''Takes all lists and returns the value for 5 test statistics:
    macro-recall, macro-precision, macro-f-score, micro-f-score, accuracy
    
    scoring: the function that calcutates the performances 
    absolute: if True : the absolute difference of system1 performance and system2 performance
              if False: system1 performance minus system2 performance
    '''
     # Get the reference performance difference
    scores1 =  scoring(gold, system1, training=training)
    scores2 =  scoring(gold, system2, training=training)

    # Compute the differences between system1 and system2
    diffs={}
    for k in set(list(scores1.keys())+list(scores2.keys())):
        diff = scores1.get(k, 0)-scores2.get(k, 0)
        if absolute: diff = abs(diff)
    
        diffs[k] = diff
    
    return diffs

    
def distribute(s):
    '''Distribute the elements of s randomly over 2 lists'''
    batch1=[]; batch2 =[]
    data=s[:]
    while data:
        d = data.pop()
        b = random.choice([batch1, batch2]) 
        b.append(d)
            
    assert len(data) == 0, data 
    
    return set(batch1), set(batch2)

def getprobabilities(ngecounts, N, add=1, verbose=False):
    '''Calculates the probabilities from the ngecounts.
    
    The probabilities are calculated as:
     (neg + add)/(N + add)
    
    Returns a dictionay:
        keys: performance name
        value: probaility

    ngecounts: a dictionary:
        keys: performance name
        value: nge
    N: number of trials
    add: integer
    '''
    # Calculate probabilities
    probs={}
    for k, nge in list(ngecounts.items()):
        prob = (nge + add)/float(N + add)
        probs[k] = prob

    if verbose:
        print('Probabilities for accepting H0:')
        names = list(probs.keys())
        names.sort()
        for name in names:
            print(('  %-23s: %.5g' %(name, probs[name])))

    return probs


def get_alternatives(l):
    '''The length of the outputs'''
    # number of bins
    nbins = int(pow(2, l))

    # Fill the bins
    bins=[[] for i in range(nbins)]    
    for i in range(l):
        switchpoint = pow(2, i)
        filler=False
        for j, bin in enumerate(bins):
            if not j%switchpoint:
                filler = not filler
            bin.append(int(filler))
            
    return bins
    
    


def exactlabelingsignificance(gold, system1, system2, verbose=False, training=None, scoring=getscores, common=[], common_gold=[]):
    '''Carries out exact randomization'''
    # number of permutations
    N = pow(2, len(gold))
    if verbose: loginfo('%d permutations' %N)
    if N > 5000000: raise ValueError('The number of permutations is too big. Aborting.')
    
    # the reference test statitsics
    refdiffs = teststatistic(gold.union(common_gold), system1.union(common), system2.union(common), training=training, scoring=scoring)
    
    # Get all combinations
    size = len(gold)
    count=0
    systems = [system1, system2]
    ngecounts = {}
    
    if N >= 10:
        nom = int(N/10.0)
    else:
        nom=1
    
    alternatives = get_alternatives(size)
    while alternatives:
        alt = alternatives.pop()
        count+=1

        shuffle1 = [systems[k][j] for j,k in enumerate(alt)]
        shuffle2 = [systems[1-k][j] for j,k in enumerate(alt)]
    
        # the test statistics
        diffs = teststatistic(gold+common_gold, shuffle1+common, shuffle2+common, training=training, scoring=scoring)
        
        if verbose and not (count%nom): loginfo('Calculated permutation %d/%d' %(count, N))
        
        for k in list(refdiffs.keys()):
            pseudo = diffs[k]
            actual = refdiffs[k]
            if pseudo >= actual:
                ngecounts[k] = ngecounts.get(k, 0) + 1
            elif k not in list(ngecounts.keys()):
                ngecounts[k]=0
                    
    assert count == N
    assert set(ngecounts.keys()) == set(refdiffs.keys())
        
    # Calculate probabilities
    return getprobabilities(ngecounts, N, add=0, verbose=True)


def labelingsignificance(gold, system1, system2, N=1000, verbose=False, training=None, scoring=getscores, show_probs=True, common=[], common_gold=[]):
    '''Calculate approximate randomization test for class labeling experiments
    
    Returns the probabilities for accepting H0 for
    macro-recall, macro-precision, macro-fscore, micro-fscore, accuracy
    
    training: the counts of the class labels in the training file
    N: number of iterations
    '''
    # the reference test statitsics
    refdiffs = teststatistic(gold+common_gold, system1+common, system2+common, training=training, scoring=scoring)
    
    # start shuffling
    source = [[s1,s2] for s1,s2 in zip(system1, system2)]
    
    if N >= 10:
        nom = int(N/10.0)
    else:
        nom=1
        
    ngecounts={}
    for i in range(N):
        shuffle1=[]
        shuffle2=[]
        for preds in source:
            random.shuffle(preds)
            shuffle1.append(preds[0])
            shuffle2.append(preds[1])
            
        # the test statistics
        diffs = teststatistic(gold+common_gold, shuffle1+common, shuffle2+common, training=training, scoring=scoring)
        
        # see whether the shuffled system performs better than the originals
        for k in list(refdiffs.keys()):
            pseudo = diffs[k]
            actual = refdiffs[k]
            if pseudo >= actual:
                ngecounts[k] = ngecounts.get(k, 0) + 1
            elif k not in list(ngecounts.keys()):
                ngecounts[k]=0
                
        if verbose and not ((i+1)%nom):
            loginfo('Calculated shuffle %d/%d' %(i+1, N))
            #getprobabilities(ngecounts, i+1, add=1, verbose=True)
    
    # Sign test check
    if scoring.__name__ == 'getscores':
        try:
            s = signtest(gold, system1, system2)
            if verbose: loginfo('Sign-test probability: %.4g' %s)
        except NameError:
            pass

    assert set(ngecounts.keys()) == set(refdiffs.keys())

    # Calculate probabilities
    return getprobabilities(ngecounts, N, add=1, verbose=show_probs)
    
   
def exacttermsignificance(gold, system1, system2, verbose=False, absolute=False):
    '''Compute exact term significance'''
    # Take unique terms 
    source = []
    doubles=[]
    for t in system1.union(system2):
        if t in system1 and t not in system2:
            source.append(t)
        elif t not in system1 and t in system2:
            source.append(t)
        else:
            doubles.append(t)
    
    # The number of combinations
    N=1
    for i in range(len(source)+1):
        N += combinations.ncombinations(len(source), i)
    if verbose:
        loginfo('%d combinations' % N)
    if N > 5000000:
        raise ValueError('The number of permutations is too big. Aborting.')
       
    # the reference test statitsics
    refdiffs = teststatistic(gold, system1, system2, scoring=getscores2, absolute=absolute)

    if N >= 10:
        nom = int(N/10.0)
    else:
        nom=1
               
    count=0
    ngecounts={}
    for i in range(len(source)+1):
       for subset in combinations.subsets(source, i):
            count+=1
            
            shuffle1 = list(subset)
            shuffle2 = []
            for x in source:
                if x not in shuffle1:
                    shuffle2.append(x)
            
            #print shuffle1, shuffle2, doubles
            
            # the test statistics
            diffs = teststatistic(gold, set(shuffle1+doubles), set(shuffle2+doubles), scoring=getscores2, absolute=absolute)
       
            # see whether the shuffled system performs better than the originals
            for k in list(refdiffs.keys()):
                pseudo = diffs[k]
                actual = refdiffs[k]
                if pseudo >= actual:
                    ngecounts[k] = ngecounts.get(k, 0) + 1
                elif k not in list(ngecounts.keys()):
                    ngecounts[k]=0
                
            if verbose and not ((count)%nom):
                loginfo('Calculated combination %d/%d' %(count, N))
                #getprobabilities(ngecounts, i+1, add=1, verbose=verbose)
        
    #assert count == N
    assert set(ngecounts.keys()) == set(refdiffs.keys())
        
    # Calculate probabilities
    probs = getprobabilities(ngecounts, N, add=0, verbose=True)

    return probs
       
def termsignificance(gold, system1, system2, N=10000, verbose=False, absolute=False):
    '''Calculate randomized term significance'''
    # Only uniques terms in a system
    assert len(gold) == len(gold)
    assert len(system1) == len(system1)
    assert len(system2) == len(system2)
    
    # Get all terms that are unique for a system
    source = []
    doubles=[]
    news1=[]; news2=[]
    for t in system1.union(system2):
        if t in system1 and t not in system2:
            source.append(t)
            news1.append(t)
        elif t not in system1 and t in system2:
            source.append(t)
            news2.append(t)
        else:
            doubles.append(t)
    
    # the reference test statitsics
    refdiffs = teststatistic(gold, system1, system2, scoring=getscores2, absolute=absolute)
    if N >= 10:
        nom = int(N/10.0)
    else:
        nom=1

    ngecounts={}
    for i in range(N):
        shuffle1, shuffle2 = distribute(source)

        # the test statistics
        diffs = teststatistic(gold, shuffle1.union(doubles), shuffle2.union(doubles), scoring=getscores2, absolute=absolute)

        # see whether the shuffled system performs better than the originals
        for k in list(refdiffs.keys()):
            pseudo = diffs[k]
            actual = refdiffs[k]
            if pseudo >= actual:
                ngecounts[k] = ngecounts.get(k, 0) + 1
            elif k not in list(ngecounts.keys()):
                ngecounts[k]=0
            
        if not ((i+1)%nom):
            loginfo('Calculated shuffle %d/%d' %(i+1, N))
            #getprobabilities(ngecounts, i+1, add=1, verbose=verbose)
    assert set(ngecounts.keys()) == set(refdiffs.keys())
    
    # Calculate probabilities
    probs = getprobabilities(ngecounts, N, add=1, verbose=True)
    return probs
        
        
        
def getdifference(system1, system2, gold=None):
    '''
    Takes lists of labels and returns lists with only those
    entries for which s1!=s2.
    
    If the list gold is given, it also returns only the gold labels
    for those elements.
    '''
    new_system1=[]
    new_system2=[]
    new_gold=[]
    
    rest1=[]
    rest2=[]
    common_gold=[]
    
    G=True
    if gold is None:
        G=False
        gold = system1[:]
    
    if len(system1) != len(system1) != len(gold): 
        raise ValueError('Input lists should have the same length')
    
    for g, s1, s2 in zip(gold, system1, system2):
        if s1!=s2:
            new_system1.append(s1)
            new_system2.append(s2)
            
            if G:
                new_gold.append(g)
        else:
            rest1.append(s1)
            rest2.append(s2)
            common_gold.append(g)
    
    if not G: new_gold=[]
    
    assert rest1 == rest2
    
    return new_system1, new_system2, new_gold, rest1, common_gold 
    
        

def main(gold, system1, system2, verbose=False, N=10000, exact_threshold=20, training=None, scoring=getscores):
    print ("IN MAIN 0\n");

    '''
    exact_threshold: the maximum number of instance to calculate exact randomization instead of approximate
    '''
    # Check
    if not (len(gold) == len(system1) == len(system2)):
        raise ValueError('There should be an equal number of non-empty lines in each input file.')
    
    # Shuffle only those instances that have a different class label    
    news1, news2, newgold, common, common_gold = getdifference(system1, system2, gold)

    if verbose:
        for i,s in enumerate([system1, system2]):
            scores = scoring(gold, s, training=training)
            
            lines=['Scores for system%d:' %(i+1)]
            keys = list(scores.keys())
            keys.sort()
            for k in keys:
                lines.append('  %-23s : %.4f' %(k, scores[k]))
            
            print(('\n'.join(lines)))
    
    # only shuffle difference: quicker and same probability results
    gold = newgold
    system1 = news1
    system2 = news2

    total_uniq = len(gold)
    
    # The number of instances with different predictions
    if verbose: loginfo('Found %d predictions that are different for the 2 systems' %(total_uniq))

    # number of permutations
    try:
        np = pow(2, len(gold))
    except OverflowError:
        np = 1000000001
    
    if np > 1000000000:
        loginfo('Number of permutations: more than 1,000,000,000')
    else:
        loginfo('Number of permutations: %d' %np)
    if np <= N and total_uniq > exact_threshold:
        loginfo('NOTE:')
        loginfo('The number of permutations is lower than the number of shuffles.')
        loginfo('You may want to calculate exact randomization. To do this')
        loginfo('set option -t higher than %d.' %total_uniq)
        
    
    if total_uniq <= exact_threshold:
        if verbose: loginfo('This is equal or less than the %d predictions threshold: calculating exact randomization' %(exact_threshold))
        probs = exactlabelingsignificance(gold, system1, system2, verbose=verbose, training=training, scoring=scoring, common=common, common_gold=common_gold)
    else:
        probs = labelingsignificance(gold, system1, system2, N=N, verbose=verbose, training=training, scoring=scoring, common=common, common_gold=common_gold)

    if verbose: loginfo('Done')

    return probs


def main2(gold, system1, system2, verbose=False, N=1048576, absolute=True, exact_threshold=10):
    ''' the main for term extraction'''
    # No doubles
    #news1 = list(set(system1))
    #news2 = list(set(system2))
    #newgold = list(set(gold))
    news1 = set(system1)
    news2 = set(system2)
    newgold = set(gold)
    gold = newgold
    system1 = news1
    system2 = news2

    if verbose:
        for i,s in enumerate([system1, system2]):
            scores = getscores2(gold, s, training=training)
            
            lines=['Scores for system%d:' %(i+1)]
            keys = list(scores.keys())
            keys.sort()
            for k in keys:
                lines.append('  %-23s : %.4f' %(k, scores[k]))
            
            print(('\n'.join(lines)))


    # the number of terms that occur only in s1 or in s2
    union=system1.union(system2)
    intersect = system1.intersection(system2)
    total_uniq = len(union) - len(intersect)

    if verbose: loginfo('Found %d predictions that are different for the 2 systems' %(total_uniq))
    
    if total_uniq < exact_threshold:
        if verbose: loginfo('This is equal of less than the %d terms threshold: calculating exact randomization' %(exact_threshold))
        probs = exacttermsignificance(gold, system1, system2, verbose=verbose, absolute=absolute)
    else:
        probs= termsignificance(gold, system1, system2, N=N, verbose=verbose, absolute=absolute)
    
    if verbose: loginfo('Done')
     
    return probs


def main3(data, verbose=False, N=1048576, absolute=True):
    '''For stratified shuffling'''
    # The groups
    scoring_func=average
    groups = list(data[list(data.keys())[0]].keys())
    groups.sort()
    assert len(groups) == 2
    
    if verbose:
        strata = list(data.keys())
        strata.sort()
        stext = 'a'
        if len(strata) == 1: stext='um'
        loginfo('Found %d strat%s: %s' %(len(data), stext, ', '.join(strata)))
        loginfo('')
        loginfo('Computing %d shuffles' %N)
        loginfo('H0: there is no absolute difference between the means of %s and %s' %tuple(groups))
    
        loginfo('    Commonly, you reject H0 if the probability drops below')
        loginfo('    a predefined significance level, e.g 0.05.')
        loginfo('-'*50)
    
    systems={groups[0]:[], groups[1]:[]}
    for stratum, d in list(data.items()):
        for g in groups:
            systems[g] += d[g]
    
    if verbose:
        for g in groups:
            s = systems[g]
            scores = scoring_func(None, s)
            
            lines=['Scores for group %s:' %(g)]
            keys = list(scores.keys())
            keys.sort()
            for k in keys:
                lines.append('  %-23s : %.4f' %(k, scores[k]))
            
            print(('\n'.join(lines)))

    # Reference
    refdiffs = teststatistic(None, systems[groups[0]], systems[groups[1]], training=None, scoring=average, absolute=absolute)

    if N >= 10:
        nom = N // 10
    else:
        nom=1

    # Start shuffling
    ngecounts={}
    for i in range(N):
        shuffled={}
        for stratum, d in list(data.items()):
            values = d[groups[0]] + d[groups[1]]
            n1 = len(d[groups[0]]) 
            n2 = len(d[groups[1]])
            labels = [groups[0]]*n1+ [groups[1]]*n2 

            random.shuffle(labels)
        
            for l, v in zip(labels, values):
                shuffled[l] = shuffled.get(l ,[]) + [v]
        
        # the test statistics
        diffs = teststatistic(None, shuffled[groups[0]], shuffled[groups[1]], scoring=scoring_func, absolute=absolute)

        # see whether the shuffled system performs better than the originals
        for k in list(refdiffs.keys()):
            pseudo = diffs[k]
            actual = refdiffs[k]
            
            if pseudo >= actual:
                ngecounts[k] = ngecounts.get(k, 0) + 1
            elif k not in list(ngecounts.keys()):
                ngecounts[k] = 0
                
        if verbose and not ((i+1)%nom):
            loginfo('Calculated shuffle %d/%d' %(i+1, N))

    assert set(ngecounts.keys()) == set(refdiffs.keys())
        
    # Calculate probabilities
    probs = getprobabilities(ngecounts, N, add=1, verbose=True)

    return probs


# ========================================================================================================================
# TESTING
# ========================================================================================================================


def Yeh():
    '''Creates 3 synthetic files to reproduce the results from Section3.3 of
    
 Alexander Yeh, More accurate tests for the statistical significance of result differences,
 in: Proceedings of the 18th International Conference on Computational Linguistics, Volume 2,
 pages 947-953, 2000.
 
The filenames are yeh.gold, yeh.s1 and yeh.s2
 
Running the following command reproduces the reported results:
 
 $ python art.py -c yeh.gold -n1048576 -v -r -a  yeh.s1 yeh.s2
 
   Probabilities for accepting H0:
     f1-score            : 0.014643
     precision           : 0.97995
     recall              : 0.00010204

Note that the test statistic is system1-system2, so for precision the 
probability from Yeh is 1 - 0.97995 = 0.02005
'''
    gold = 'yeh.gold'
    s1 = 'yeh.s1'
    s2 = 'yeh.s2'
    # The gold standard
    with open(gold, 'w') as f:
        for i in range(103):
            f.write('%d\n' %i)
    
    # System 1: R45.6 P49.5 F47.5
    with open(s1, 'w') as f:
        for i in range(19+28):
            f.write('%d\n' %i) # retrieved by both and system1
        for i in range(5):
            f.write('B%d\n' %(i)) # spurious retrieved by both
        for i in range(43):
            f.write('one%d\n' %(i)) # spurious retrieved by system1
        
    # System 2: R24.3 P64.1 F35.2
    with open(s2, 'w') as f:
        for i in range(19+6):
            if i < 19:
                f.write('%d\n' %i) # retrieved by both
            else:
                f.write('%d\n' %(i+28)) # retrieved by system2
        for i in range(5):
            f.write('B%d\n' %(i)) # spurious retrieved by both
        for i in range(9):
            f.write('two%d\n' %(i)) # spurious retrieved by system1
        
    print(('Written:', gold, s1, s2))

# ==================================================================================================================

try:
    opts,args=getopt.getopt(sys.argv[1:],'hHc:s:vn:t:T:ram', ['help', 'yeh'])
except getopt.GetoptError:
    # print help information and exit:
    print('o no')
    sys.exit(2)

sep=None
gold = None
verbose=False
N=10000
exact_threshold=10
trainingfile = None
training=None
terms=False
absolute=True
mbt=False

for o, a in opts:
    if o in ('-h', '--help'):
        sys.exit()
    if o in ('-H',):
        sys.exit(2)
    if o in ('-s',):
        sep = a
        if sep == '\\t': sep='\t'
    if o in ('-c',):
        gold = a
    if o in ('-v',):
        verbose = True
    if o in ('-n',):
        N = int(a)
    if o in ('-t',):
        exact_threshold = int(a)
    if o in ('-T',):
        trainingfile = a
    if o in ('-r',):
        terms = True
    if o in ('-a',):
        absolute = False
    if o in ('-m',):
        mbt = True
    if o in ('--yeh',):
        Yeh()
        sys.exit(0)

if len(args) == 1:
    data = strata_read(args[0], sep=sep)
    loginfo('-'*50)
    loginfo('Datafile: %s' %os.path.basename(args[0]))
    main3(data, verbose=verbose, N=N)
    sys.exit(0)
elif len(args) != 2:
    sys.exit(1)

# The files with the systems
output1, output2 = args


if terms and not gold:
    print('ERROR 2: when doing term significance testing a gold standard is needed (-c option)')
    sys.exit(1)

if mbt and not gold:
    print('ERROR 3: when doing MBT significance testing a gold standard is needed (-c option)')
    sys.exit(1)

# Reading in the class labels
if gold:
    if mbt:
        goldlabels = mbtread(gold)
        system1 = mbtread(output1)
        system2 = mbtread(output2)
    else:
        if trainingfile: training = readtraining(trainingfile, sep=sep, index=None)

        goldlabels = fread(gold, index=None)
        system1 =  fread(output1, index=None)
        system2 =  fread(output2, index=None)
else:
    if trainingfile: training = readtraining(trainingfile, sep=sep, index=-1)

    try:
        goldlabels = fread(output1, index=-2, sep=sep)
    except IndexError:
        print(('ERROR 4: Is the feature separator set correctly? (option -s is currently "%s")' %str(sep)))
        sys.exit(1)
    check = fread(output2, index=-2, sep=sep)

    if check != goldlabels:
        print((check, goldlabels))
        print(('ERROR 5: File %s and %s should have the same gold reference labels.' %(output1, output2)))
        sys.exit(1)
    del check

    check1 = fread(output1, index=(0,-1), sep=sep)
    check2 = fread(output2, index=(0,-1), sep=sep)

    if check1 != check2:
        print(('ERROR 5: File %s and %s should be exactly the same up until the predicted class label.' %(output1, output2)))
        sys.exit(1)
    del check1, check2

    system1=fread(output1, index=-1, sep=sep)
    system2=fread(output2, index=-1, sep=sep)

# Info
if verbose:
    loginfo('-'*50)
    loginfo('SYSTEM1 :%s' %output1)
    loginfo('SYSTEM2 :%s' %output2)
    if mbt:
        loginfo('GOLD    :%s' %gold)
        loginfo('MBT style formatted files')
        loginfo('%d sentences in input files' %len(system1))
    else:
        if gold:
            loginfo('GOLD    :%s' %gold)
            if not terms: loginfo('Considering entire lines as class labels')
        else:
            loginfo('Considering the last field as the predicted class label')
            loginfo('Considering the one but last field as the gold standard class label')
            if sep is not None: loginfo('Using "%s" as feature separator' %sep)
        if not terms: loginfo('%d instances in input files' %len(system1))

    labels=set(goldlabels)
    labels = labels.union(set(system1))
    labels = labels.union(set(system2))
    nlabels = len(labels)
    labels=list(labels)
    labels.sort()

    if not mbt: loginfo('Found %d different labels/terms' %nlabels)
    if nlabels < 10: loginfo('  %s' %(', '.join(labels)))

    if trainingfile: loginfo('Computing averaged scores using class label counts from: %s' %trainingfile)

    loginfo('')
    loginfo('Computing %d shuffles' %N)
    loginfo('H0: there is no difference between SYSTEM1 and SYSTEM2')
    if terms and not absolute: loginfo('H1: SYSTEM1 performs better than SYSTEM2')

    loginfo('    Commonly, you reject H0 if the probability drops below')
    loginfo('    a predefined significance level, e.g 0.05.')
    loginfo('-'*50)




if gold and mbt:
    probs = main(goldlabels, system1, system2, verbose=verbose, N=N, exact_threshold=exact_threshold, training=None, scoring=getscoresmbt)
    #probs = main(goldlabels, system1, system2, verbose=verbose, N=N, exact_threshold=exact_threshold, training=None, scoring=getscoresmbtmulti)
elif gold and terms:
    probs = main2(goldlabels, system1, system2, N=N, verbose=verbose, absolute=absolute, exact_threshold=exact_threshold)
else:
    probs = main(goldlabels, system1, system2, verbose=verbose, N=N, exact_threshold=exact_threshold, training=training) #, scoring=getscoresmbtmulti)

print('done')