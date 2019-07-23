#!/usr/local/bin/python
import argparse
import glob
import os
from collections import defaultdict
#from xml.etree import cElementTree
#from art import *

"""Significance Test Calculator."""

"""
To run this file, please use:

python <gold standard folder> <system 1 output folder> <system 2 output folder>

e.g.: python gold_annotations system1_annotations system2_annotations

Please note that you must use Python3 to get the correct results with this script

Creates 3 file of annotationIDs on each line. The gold, system1, system2. Assigns a unique ID to each concept/relation in the gold. The system files contain each matches gold ID, any matched to other system IDs (but not in gold), and any unique IDs. The result is a list of matched gold, matched system not in gold, and matched neither system or gold for each system. These are then input into the significance test 

Much of the code is copied from Track2-evaluate-ver4.py 

"""

class ClinicalConcept(object):
    """Named Entity Tag class."""
    def __init__(self, tid, start, end, ttype, text=''):
        """Init."""
        self.tid = str(tid).strip()
        self.start = int(start)
        self.end = int(end)
        self.text = str(text).strip()
        self.ttype = str(ttype).strip()

    def span_matches(self, other, mode='strict'):
        """Return whether the current tag overlaps with the one provided."""
        assert mode in ('strict', 'lenient')
        if mode == 'strict':
            if self.start == other.start and self.end == other.end:
                return True
        else:   # lenient
            if (self.end > other.start and self.start < other.end) or \
               (self.start < other.end and other.start < self.end):
                return True
        return False

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        return other.ttype == self.ttype and self.span_matches(other, mode)

    def __str__(self):
        """String representation."""
        return '{}\t{}\t({}:{})'.format(self.ttype, self.text, self.start, self.end)


class Relation(object):
    """Relation class."""
    def __init__(self, rid, arg1, arg2, rtype):
        """Init."""
        assert isinstance(arg1, ClinicalConcept)
        assert isinstance(arg2, ClinicalConcept)
        self.rid = str(rid).strip()
        self.arg1 = arg1
        self.arg2 = arg2
        self.rtype = str(rtype).strip()
        
    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        if self.arg1.equals(other.arg1, mode) and \
                self.arg2.equals(other.arg2, mode) and \
                self.rtype == other.rtype:
            return True
        return False

    def __str__(self):
        """String representation."""
        return '{} ({}->{})'.format(self.rtype, self.arg1.ttype,
                                    self.arg2.ttype)


class RecordTrack2(object):
    """Record for Track 2 class."""

    def __init__(self, file_path):
        """Initialize."""
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()
        #self.text = self._get_text()

    @property
    def tags(self):
        return self.annotations['tags']

    @property
    def relations(self):
        return self.annotations['relations']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        with open(self.path) as annotation_file:
            lines = annotation_file.readlines()
            for line_num, line in enumerate(lines):
                if line.strip().startswith('T'):
                    try:
                        tag_id, tag_m, tag_text = line.strip().split('\t')
                    except ValueError:
                        print(self.path, line)
                    if len(tag_m.split(' ')) == 3:
                        tag_type, tag_start, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 4:
                        tag_type, tag_start, _, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 5:
                        tag_type, tag_start, _, _, tag_end = tag_m.split(' ')
                    else:
                        print(self.path)
                        print(line)
                    tag_start, tag_end = int(tag_start), int(tag_end)
                    annotations['tags'][tag_id] = ClinicalConcept(tag_id,
                                                                  tag_start,
                                                                  tag_end,
                                                                  tag_type,
                                                                  tag_text)

            for line_num, line in enumerate(lines):
                if line.strip().startswith('R'):
                    rel_id, rel_m = line.strip().split('\t')
                    rel_type, rel_arg1, rel_arg2 = rel_m.split(' ')
                    rel_arg1 = rel_arg1.split(':')[1]
                    rel_arg2 = rel_arg2.split(':')[1]
                    arg1 = annotations['tags'][rel_arg1]
                    arg2 = annotations['tags'][rel_arg2]
                    annotations['relations'][rel_id] = Relation(rel_id, arg1,
                                                                arg2, rel_type)
        return annotations

    def _get_text(self):
        """Return the text in the corresponding txt file."""
        path = self.path.replace('.ann', '.txt')
        with open(path) as text_file:
            text = text_file.read()
        return text

    def search_by_id(self, key):
        """Search by id among both tags and relations."""
        try:
            return self.annotations['tags'][key]
        except KeyError():
            try:
                return self.annotations['relations'][key]
            except KeyError():
                return None



class SingleEvaluator(object):
    """Evaluate two single files."""
    def __init__(self, doc1, doc2, doc3, mode='strict', key=None):
        """Initialize."""
        assert isinstance(doc1, RecordTrack2)
        assert isinstance(doc2, RecordTrack2)
        assert isinstance(doc3, RecordTrack2)
        assert mode in ('strict', 'lenient')
        assert doc1.basename == doc2.basename == doc3.basename
        self.scores = {'tags': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
                       'relations': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}}
        self.doc1 = doc1
        self.doc2 = doc2
        self.doc3 = doc3

        ####### Concepts #################################
        if key:
            golCon = {t for t in doc1.tags.values() if t.ttype == key}
            sys1Con = {t for t in doc2.tags.values() if t.ttype == key}
            sys2Con = {t for t in doc3.tags.values() if t.ttype == key}
            sys1Con_check = {t for t in doc2.tags.values() if t.ttype == key}
            sys2Con_check = {t for t in doc3.tags.values() if t.ttype == key}
        else:
            golCon = {t for t in doc1.tags.values()}
            sys1Con = {t for t in doc2.tags.values()}
            sys2Con = {t for t in doc3.tags.values()}
            sys1Con_check = {t for t in doc2.tags.values()}
            sys2Con_check = {t for t in doc3.tags.values()}

        #pare down matches -- if multiple system tags overlap with only one
        #gold standard tag, only keep one sys tag
        gol_matched = []
        for s1 in sys1Con:
            for g in golCon:
                if (g.equals(s1,mode)):
                    if g not in gol_matched:
                        gol_matched.append(g)
                    else:
                        if s1 in sys1Con_check:
                            sys1Con_check.remove(s1)
        sys1Con = sys1Con_check
        gol_matched = []
        for s2 in sys2Con:
            for g in golCon:
                if (g.equals(s2,mode)):
                    if g not in gol_matched:
                        gol_matched.append(g)
                    else:
                        if s2 in sys2Con_check:
                            sys2Con_check.remove(s2)
        sys2Con = sys2Con_check

        #create a set of all samples
        allConSamples = set()
        allConSys1 = set()
        allConSys2 = set()
        #first add all the gold samples
        for g in golCon:
            allConSamples.add(g)
            g.tid = ("Con_"+str(doc1.basename)+"_"+str(len(allConSamples)));
        #next, add any samples that are in sys1 and not gold
        for s in sys1Con:
            matched = 0
            matchedSample = None
            for a in golCon:
                if s.equals(a,mode):
                    matched = 1
                    matchedSample = a
                    break
            if (matched > 0):
                allConSys1.add(matchedSample)
            else:
                allConSys1.add(s)
                allConSamples.add(s)
                s.tid = ("Con_"+str(doc1.basename)+"_"+str(len(allConSamples)));
        #next, add any that are in sys2, but not gold or sys1
        # which is allSamples
        for s in sys2Con:
            matched = 0
            matchedSample = None
            for a in golCon:
                if s.equals(a,mode):
                    matched = 1
                    matchedSample = a
                    break
            if (matched < 1):
                for a in allConSys1:
                    if s.equals(a,mode):
                        #since there can be overlapping samples, check that its 
                        #not already in allConSys 2. For instance, sys1 tags too
                        #much, encompassing 2 of sys2 tags.
                        if (not (a in allConSys2)):
                            matched = 1
                            matchedSample = a
                            break
            if matched:
                allConSys2.add(matchedSample)
            else:
                allConSys2.add(s)
                allConSamples.add(s)
                s.tid = ("Con_"+str(doc1.basename)+"_"+str(len(allConSamples)));

        #save to self
        self.gold_c = golCon
        self.sys1_c = allConSys1
        self.sys2_c = allConSys2

    
        ####### Relations #################################
        if key:
            golRel = [r for r in doc1.relations.values() if r.rtype == key]
            sys1Rel = [r for r in doc2.relations.values() if r.rtype == key]
            sys2Rel = [r for r in doc3.relations.values() if r.rtype == key]
            sys1Rel_check = [r for r in doc2.relations.values() if r.rtype == key]
            sys2Rel_check = [r for r in doc3.relations.values() if r.rtype == key]
        else:
            golRel = [r for r in doc1.relations.values()]
            sys1Rel = [r for r in doc2.relations.values()]
            sys2Rel = [r for r in doc3.relations.values()]
            sys1Rel_check = [r for r in doc2.relations.values()]
            sys2Rel_check = [r for r in doc3.relations.values()]

        #pare down matches -- if multiple system tags overlap with only one
        #gold standard tag, only keep one sys tag
        gol_matched = []
        for s1 in sys1Rel:
            for g in golRel:
                if (g.equals(s1,mode)):
                    if g not in gol_matched:
                        gol_matched.append(g)
                    else:
                        if s1 in sys1Rel_check:
                            sys1Rel_check.remove(s1)
        sys1Rel = sys1Rel_check
        gol_matched = []
        for s2 in sys2Rel:
            for g in golRel:
                if (g.equals(s2,mode)):
                    if g not in gol_matched:
                        gol_matched.append(g)
                    else:
                        if s2 in sys2Rel_check:
                            sys2Rel_check.remove(s2)
        sys2Rel = sys2Rel_check

        #create a set of all samples
        allRelSamples = set()
        allRelSys1 = set()
        allRelSys2 = set()
        for g in golRel:
            allRelSamples.add(g)
            g.rid = ("REL_"+str(doc1.basename)+"_"+str(len(allRelSamples)));
        for s in sys1Rel:
            matched = 0
            matchedSample = None
            for a in golRel:
                if s.equals(a,mode):
                    matched = 1
                    matchedSample = a
                    break
            if (matched > 0):
                allRelSys1.add(matchedSample)
            else:
                allRelSys1.add(s)
                allRelSamples.add(s)
                s.rid = ("REL_"+str(doc1.basename)+"_"+str(len(allRelSamples)));

        for s in sys2Rel:
            matched = 0
            matchedSample = None
            for a in golRel:
                if s.equals(a,mode):
                    matched = 1
                    matchedSample = a
                    break
            if (matched < 1):
                for a in allRelSys1:
                    if s.equals(a,mode):
                        #since there can be overlapping samples, check that its 
                        #not already in allRelSys 2. For instance, sys1 tags too
                        #much, encompassing 2 of sys2 tags.
                        if (not (a in allRelSys2)):
                            matched = 1
                            matchedSample = a
                            break
            if (matched > 0):
                allRelSys2.add(matchedSample)
            else:
                allRelSys2.add(s)
                allRelSamples.add(s)
                s.rid = ("REL_"+str(doc1.basename)+"_"+str(len(allRelSamples)));
        #save to self
        self.gold_r = golRel
        self.sys1_r = allRelSys1
        self.sys2_r = allRelSys2

class MultipleEvaluator(object):
    """Evaluate two sets of files."""

    def __init__(self, corpora, mode='strict', tag_type=None):
        """Initialize."""
        assert isinstance(corpora, Corpora)
        assert mode in ('strict', 'lenient')
        self.track2(corpora, tag_type, mode)

    def track2(self, corpora, tag_type=None, mode='strict'):
        #for each file, count the numGold, in Both not in gold, system 1 only, and system 2 only
        self.gold_c = set()
        self.sys1_c = set()
        self.sys2_c = set()
        self.gold_r = set()
        self.sys1_r = set()
        self.sys2_r = set()
        for g, s1, s2 in corpora.docs:
            evaluator = SingleEvaluator(g, s1, s2, mode, tag_type)
            self.sys1_c = self.sys1_c.union(evaluator.sys1_c)
            self.sys2_c = self.sys2_c.union(evaluator.sys2_c)
            self.gold_c = self.gold_c.union(evaluator.gold_c)
            self.sys1_r = self.sys1_r.union(evaluator.sys1_r)
            self.sys2_r = self.sys2_r.union(evaluator.sys2_r)
            self.gold_r = self.gold_r.union(evaluator.gold_r)
       

def evaluate(corpora, outDir, mode='strict'):
    """Run the evaluation by considering only files in the two folders."""
    assert mode in ('strict', 'lenient')
    evaluator = MultipleEvaluator(corpora, mode)
    
    ######
    # ouptut the results to the output directory for concepts
    #goldFile for concepts
    fileName = outDir + "/gold_c"
    outFile = open(fileName,'w')
    for g in evaluator.gold_c:
        outFile.write(str(g.tid) + "\n")
    outFile.close()

    #sys1File for concepts
    fileName = outDir + "/sys1_c"
    outFile = open(fileName,'w')
    for s in evaluator.sys1_c:
        outFile.write(str(s.tid)+"\n")
    outFile.close();

    #sys2File for concepts
    fileName = outDir + "/sys2_c"
    outFile = open(fileName,'w')
    for s in evaluator.sys2_c:
        outFile.write(str(s.tid)+"\n")
    outFile.close();

    # output the results to the output directory for relations
    #goldFile for relations
    fileName = outDir + "/gold_r"
    outFile = open(fileName,'w')
    for g in evaluator.gold_r:
        outFile.write(str(g.rid) + "\n")
    outFile.close()

    #sys1File for concepts
    fileName = outDir + "/sys1_r"
    outFile = open(fileName,'w')
    for s in evaluator.sys1_r:
        outFile.write(str(s.rid)+"\n")
    outFile.close();

    #sys2File for concepts
    fileName = outDir + "/sys2_r"
    outFile = open(fileName,'w')
    for s in evaluator.sys2_r:
        outFile.write(str(s.rid)+"\n")
    outFile.close();

    #run the significance script
        

        
class Corpora(object):

    def __init__(self, folder1, folder2, folder3):
        file_ext = '*.ann'
        self.folder1 = folder1
        self.folder2 = folder2
        self.folder3 = folder3
        files1 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder1, file_ext))])
        files2 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder2, file_ext))])
        files3 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder3, file_ext))])
        common_files = files1 & files2 & files3    # intersection
        if not common_files:
            print('ERROR: None of the files match.')
        else:
            if files1 - common_files:
                print('Files skipped in {}:'.format(self.folder1))
                print(', '.join(sorted(list(files1 - common_files))))
            if files2 - common_files:
                print('Files skipped in {}:'.format(self.folder2))
                print(', '.join(sorted(list(files2 - common_files))))
            if files3 - common_files:
                print('Files skipped in {}:'.format(self.folder3))
                print(', '.join(sorted(list(files3 - common_files))))
        self.docs = []
        for file in common_files:
            g = RecordTrack2(os.path.join(self.folder1, file))
            s1 = RecordTrack2(os.path.join(self.folder2, file))
            s2 = RecordTrack2(os.path.join(self.folder3, file))
            self.docs.append((g, s1, s2))


def main(f1, f2, f3, outDir, mode='strict'):
    """Where the magic begins."""
    corpora = Corpora(f1, f2, f3)
    if corpora.docs:
        evaluate(corpora, outDir, mode)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='n2c2: Evaluation script for Track 2')
    parser.add_argument('folder1', help='First data folder path (gold)')
    parser.add_argument('folder2', help='Second data folder path (system1)')
    parser.add_argument('folder3', help='Third data folder path (system2)')
    parser.add_argument('outDir', help='Fourth data folder path (output)')
    args = parser.parse_args()
    main(os.path.abspath(args.folder1), os.path.abspath(args.folder2), os.path.abspath(args.folder3), os.path.abspath(args.outDir),'lenient')
