#script to compute approximate randomization significances for
# each system run against every other system run. Each systen
# comparison is run in parallel via a nohup command, and results
# are output to a nohup folder in the corresponding directory
# all results can be summarized as a table using the 
# collectResults.pl script
#
# Requires 6 hardcoded values:
#   $goldFolder - folder containing gold standard annotations
#   $systemsFolder - folder containing system outputs for each system. This 
#                    should be a folder of folders, where each folder contains 
#                    the system output annotation files
#   $differencesDir - the directory to output the files needed for significance 
#                     testing, and where to run significance testing
#   $numShuffles - the number of shuffles to perform when performing approximate
#                  randomization. See that code (art.py) for more details. 
#                  Sugested default is 5000, for their work, they use a much
#                  larger number (1048576), but this takes a long time with 
#                  many samples.
#   $getConceptsDiff - 1 or 0, 1 to calculate signidicance for concept scores
#   $getRelationsDiff - 1 or 0, 1 to calculate significance for relation scores
use strict;
use warnings;


#Demo code for generating results
my $goldFolder = 'sampleData/goldStandard';
my $systemsFolder = 'sampleData/submittedRuns';
my $differencesDir = 'demo_run';
my $numShuffles = 50000;
my $getConceptsDiff = 1;
my $getRelationsDiff = 1;
&_getSignificances($goldFolder, $systemsFolder, $differencesDir, $numShuffles, $getConceptsDiff, $getRelationsDiff);



#Code for generating results for n2c2 2018 task 2 overview paper
=comment
my $goldFolder = '/home/henryst/significanceTesting_task2/gold-standard-test-data/';
my $systemsFolder = '/home/henryst/significanceTesting_task2/top10_concepts';
my $differencesDir = 'differences_concepts';
my $numShuffles = 50000;
&_getSignificances($goldFolder, $systemsFolder, $differencesDir, $numShuffles, 1, 0);

$goldFolder = '/home/henryst/significanceTesting_task2/gold-standard-test-data/';
$systemsFolder = '/home/henryst/significanceTesting_task2/top10_relations';
$differencesDir = 'differences_relations';
$numShuffles = 50000;
&_getSignificances($goldFolder, $systemsFolder, $differencesDir, $numShuffles, 0, 1);

$goldFolder = '/home/henryst/significanceTesting_task2/gold-standard-test-data/';
$systemsFolder = '/home/henryst/significanceTesting_task2/top10_endToEnd';
$differencesDir = 'differences_endToEnd';
$numShuffles = 50000;
&_getSignificances($goldFolder, $systemsFolder, $differencesDir, $numShuffles, 1, 1);
=cut




#######################################################
#      Begin Code
#######################################################

#routine to get significances for each system output
# against every other system output.
sub _getSignificances {
    my $goldFolder = shift;
    my $systemsFolder = shift;
    my $differencesDir = shift;
    my $numShuffles = shift;
    my $getConceptDiffs = shift;
    my $getRelationDiffs = shift;

    #get the folders of the system output directory
    opendir DIR, $systemsFolder or die $!;
    my @folderContents = readdir(DIR);
    closedir(DIR);
    my @systemFolders = (); 
    my @teamNames = ();
    foreach my $file (@folderContents) {
	if ($file eq '.' || $file eq '..') {
	    next;
	}
	my $path = $systemsFolder.'/'.$file;
	if (-d $path) {
	    push @systemFolders, $path;
	    push @teamNames, $file;
	}
    }
    
    #get significances for each system against each other system
    my %significances = ();
    `mkdir $differencesDir`;
    for (my $i = 0; $i < scalar @systemFolders; $i++) {
	for(my $j = 0; $j < scalar @systemFolders; $j++) {
	    #generate the score files
	    my $runDirectory = $differencesDir.'/'.$teamNames[$i].'_'.$teamNames[$j];
	    `mkdir $runDirectory`;
	    `python3 createSignificanceTestFiles.py $goldFolder $systemFolders[$i] $systemFolders[$j] $runDirectory`;
	    if ($getConceptDiffs) {
		`nohup python art.py -c $runDirectory/gold_c -n$numShuffles -v -r -a  $runDirectory/sys1_c $runDirectory/sys2_c > $runDirectory/nohup_concepts 2>&1 &`;
	    }
	    if ($getRelationDiffs) {
		`nohup python art.py -c $runDirectory/gold_r -n$numShuffles -v -r -a  $runDirectory/sys1_r $runDirectory/sys2_r > $runDirectory/nohup_relations 2>&1 &`;
	    }
	}
    }
}
