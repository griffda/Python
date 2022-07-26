# IMPORTED LIBRARIES
#sklearn imports
import sklearn
from sklearn.metrics import mean_squared_error

import csv

#libpgm imports
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.tablecpdfactorization import TableCPDFactorization
from libpgm.pgmlearner import PGMLearner

import io
import copy
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import os
import operator
import re
import networkx as nx



def discrete_estimatebn( learner, data, skel, pvalparam=.05, indegree=0.5):
    assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."

    # learn graph skeleton
    #skel = self.discrete_constraint_estimatestruct(data, pvalparam=pvalparam, indegree=indegree)

    # learn parameters
    bn = learner.discrete_mle_estimateparams(skel, data)

    # return
    return bn

def alphanum_key(s):
    key = re.split(r"(\d+)", s)
    key[1::2] = map(int, key[1::2])
    return key

def len_csvdata(csv_file_path):
    data = []

    with io.open(csv_file_path, 'r') as f:
        reader = csv.reader(f, dialect=csv.excel)

        for row in reader:
            data.append(row)

    length = len(data)
    return length

def loadDataFromCSV (csv_file_path, header=False):
    #TODO: should rewrite this function as loaddataset_kfold and write the kfold code in here and return list of lists of indexes
    dataset = []
    with open(csv_file_path, 'r') as csvfile:
        lines = csv.reader(csvfile)

        for row in lines:
            dataset.append(row)
    data = []
    if (header==True): data.append(dataset[0])
    for i in range(0, len(dataset)):
        row = []
        for j in range (0, len(dataset[i])):
            if i==0: row.append(dataset[i][j])
            else:
                item = float(dataset[i][j])
                row.append(item)
        data.append(row)
    #print(np.array(data).astype(np.float))
    #print data
    return data

def ranges(data):
    assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."
    cdata = copy.deepcopy(data)
    # establish ranges
    ranges = dict()
    for variable in cdata[0].keys():
        ranges[variable] = [float("infinity"), float("infinity") * -1]
    for sample in cdata:
        for var in sample.keys():
            if sample[var] < ranges[var][0]:
                ranges[var][0] = round(sample[var], 1)
            if sample[var] > ranges[var][1]:
                ranges[var][1] = round(sample[var], 1)

    return ranges

def bins(max, min, numBins):
    bin_ranges = []
    increment = (max - min) / float(numBins)

    for i in range(numBins - 1, -1, -1):
        a = round(max - (increment * i), 20)

        b = round(max - (increment * (i + 1)), 20)
        ##print 'b', b
        ##print 'a', a

        bin_ranges.append([b, a])
    #print bin_ranges
    return bin_ranges

def percentile_bins(array, numBins):
    a = np.array(array)

    percentage = 100.0 / numBins
    bin_widths = [0]
    bin_ranges = []
    for i in range(0, numBins):
        p_min = round ((np.percentile(a, (percentage * i))),20)
        #print 'p_min ', p_min
        bin_widths.append(p_min)
        p_max = round((np.percentile(a, (percentage * (i + 1)))), 20)
        #print 'p_max ', p_max
        bin_ranges.append([round(p_min, 20), round(p_max, 20)])


    # print bin_ranges


    #plt.hist(a, bins=bin_widths)

    #plt.show()
    return bin_ranges

def draw_barchartpd(binranges, probabilities):

    """
    combined =[]
    for range in binranges:
        for val in range:
            combined.append(val)
    # Convert to a set and back into a list.
    print combined
    sett = set(combined)
    xticks = list(sett)
    xticks.sort()
    """

    xticksv = []
    widths = []
    edge = []
    #edge.append(binranges[0][len(binranges[0])-1])
    for index, range in enumerate(binranges):
        print('range '), range
        edge.append(range[0])
        widths.append(range[1]-range[0])
        xticksv.append(((range[1]-range[0])/2)+range[0])
        if index ==len(binranges)-1: edge.append(range[1])

    print('xticks '), xticksv
    print('probabilities '), probabilities
    print('edge '), edge

    b = plt.bar(xticksv, probabilities, align='center', width=widths, color='black', alpha=0.2)

    #plt.bar(xticksv, posterior, align='center', width=widths, color='red', alpha=0.2)
    #plt.xlim(edge[0], max(edge))
    plt.xticks(edge)
    plt.ylim(0, 1)
    plt.show()

    return b

def draw_histograms(df, binwidths, n_rows, n_cols, maintitle, xlabel, ylabel, displayplt = False, saveplt =False ,**kwargs ):

    fig=plt.figure(figsize=((750*n_cols)/220, (750*n_rows)/220  ), dpi=220)
    t = fig.suptitle(maintitle, fontsize=4)
    #t.set_poition(0.5, 1.05)

    #TODO: df needs to be replaced with probabilities or write bar function that returns bar ax bar(probs, x)

    i = 0
    for var_name in list(df):
        #print df
        ax=fig.add_subplot(n_rows,n_cols,i+1)


        if isinstance(binwidths, int) == True:
            #minv = min(df[var_name])
            #maxv = max(df[var_name])
            #df[var_name].hist(bins=np.arange(minv, maxv + binwidths, binwidths),ax=ax)
            print('binwidths '), binwidths

            df[var_name].hist(bins=binwidths, ax=ax, color='black')
            #df[var_name].plot(kind='kde', ax=ax, secondary_y=False, grid=None, lw=0.5 )

        else:
            df[var_name].hist(bins=binwidths[var_name],ax=ax, color='black' )
            #df[var_name].plot(kind='kde', ax=ax, secondary_y=False , grid = None, lw=0.5)

        ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round' )
        #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.set_title(var_name, fontweight="bold", size=6)
        ax.set_ylabel(ylabel, fontsize=4)  # Y label
        ax.set_xlabel(xlabel, fontsize=4)  # X label
        ax.xaxis.set_tick_params(labelsize=4)
        ax.yaxis.set_tick_params(labelsize=4)

        #ax.grid(False)
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])

        i+=1

    fig.tight_layout()  # Improves appearance a bit.
    fig.subplots_adjust(top=0.85) #white spacing between plots and title
    # if you want to set backgrond of figure to transpearaent do it here. Use facecolor='none' as argument in savefig ()
    if displayplt == True:plt.show()

    if saveplt == True: fig.savefig('/Users/zack_sutd/Dropbox/SUTD/PhD/Thesis/Phase 2/Simple_truss/Plots/'+str(maintitle)+'.png', dpi=400)

def printdist(jd, bn, normalize=True):

    x = [bn.Vdata[i]["vals"] for i in jd.scope]
    #zipover = [i / sum(jd.vals) for i in jd.vals] if normalize else jd.vals
    s = sum(jd.vals)
    zipover = [i / s for i in jd.vals] if normalize else jd.vals

    # creates the cartesian product
    k = [a + [b] for a, b in zip([list(i) for i in itertools.product(*x[::-1])], zipover)]
    df = pd.DataFrame.from_records(k, columns=[i for i in reversed(jd.scope)] + ['probability'])
    return df

def kfoldToList (indexList, csvData, header):

    list = []
    #print 'header ', header
    list.append(header)
    for i in range (0, len(indexList)):
        list.append(csvData[indexList[i]])

    return list

def kfoldToDF (indexList, dataframe):

    df = pd.DataFrame(index = range(0, len(indexList)),columns=dataframe.columns)

    for index, dfindex in enumerate(indexList):
        df.iloc[index] = dataframe.iloc[dfindex]

    return df

def without_keys(d, keys):

    return {x: d[x] for x in d if x not in keys}

def distribution_distance_error(correct_bin_locations, predicted_bin_probabilities, actual_values, bin_ranges, plot=False):

    distance_errors = []
    norm_distance_errors = []
    output_bin_means = []

    for i in range(0, len(bin_ranges)):

        max_bound = bin_ranges[i][1]
        min_bound = bin_ranges[i][0]

        output_bin_means.append(((max_bound - min_bound) * 0.5) + min_bound)

    for i in range(len(correct_bin_locations)):
        probabilities = predicted_bin_probabilities[i]
        index, value = max(enumerate(probabilities), key=operator.itemgetter(1))  # finds bin with max probability and returns it's value and index
        actual_bin = correct_bin_locations[i]  # bin containing actual value

        # distance between bin means
        # distance_error = abs(output_bin_means[predicted_bin] - output_bin_means[actual_bin])
        # OR
        # distance between actual value and bin mean
        distance_error = abs(output_bin_means[index] - actual_values[i])

        norm_distance_error = (distance_error - bin_ranges[0][0]) / (
        bin_ranges[len(bin_ranges) - 1][1] - bin_ranges[0][0])

        distance_errors.append(distance_error)
        norm_distance_errors.append(norm_distance_error*100) # remove 100 to normalise

        #print 'distance_error', distance_error
        #print 'max def value ', bin_ranges[len(bin_ranges) - 1][1]
        #print 'min def value ', bin_ranges[0][0]
        #print 'normalised distance error ', norm_distance_error

    if plot == True:
        plt.hist(norm_distance_errors, bins=15)
        plt.xlim(-1, 1)
        plt.show()

    return norm_distance_errors
    #return distance_errors

def graph_to_pdf(nodes, edges, name):
    '''
    save a plot of the Bayes net graph in pdf
    '''
    G=nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.drawing.nx_pydot.write_dot(G,name + ".dot")
    os.system("dot -Tpdf %s > %s" % (name+'.dot', name+'.pdf'))

def discrete_mle_estimateparams2(graphskeleton, data):
    '''
    Estimate parameters for a discrete Bayesian network with a structure given by *graphskeleton* in order to maximize the probability of data given by *data*. This function takes the following arguments:

        1. *graphskeleton* -- An instance of the :doc:`GraphSkeleton <graphskeleton>` class containing vertex and edge data.
        2. *data* -- A list of dicts containing samples from the network in {vertex: value} format. Example::

                [
                    {
                        'Grade': 'B',
                        'SAT': 'lowscore',
                        ...
                    },
                    ...
                ]

    This function normalizes the distribution of a node's outcomes for each combination of its parents' outcomes. In doing so it creates an estimated tabular conditional probability distribution for each node. It then instantiates a :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>` instance based on the *graphskeleton*, and modifies that instance's *Vdata* attribute to reflect the estimated CPDs. It then returns the instance. 

    The Vdata attribute instantiated is in the format seen in :doc:`unittestdict`, as described in :doc:`discretebayesiannetwork`.

    Usage example: this would learn parameters from a set of 200 discrete samples::

        import json

        from libpgm.nodedata import NodeData
        from libpgm.graphskeleton import GraphSkeleton
        from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
        from libpgm.pgmlearner import PGMLearner

        # generate some data to use
        nd = NodeData()
        nd.load("../tests/unittestdict.txt")    # an input file
        skel = GraphSkeleton()
        skel.load("../tests/unittestdict.txt")
        skel.toporder()
        bn = DiscreteBayesianNetwork(skel, nd)
        data = bn.randomsample(200)

        # instantiate my learner 
        learner = PGMLearner()

        # estimate parameters from data and skeleton
        result = learner.discrete_mle_estimateparams(skel, data)

        # output
        print json.dumps(result.Vdata, indent=2)

    '''
    assert (isinstance(graphskeleton, GraphSkeleton)), "First arg must be a loaded GraphSkeleton class."
    assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Second arg must be a list of dicts."

    # instantiate Bayesian network, and add parent and children data
    bn = DiscreteBayesianNetwork()
    graphskeleton.toporder()
    bn.V = graphskeleton.V
    bn.E = graphskeleton.E
    bn.Vdata = dict()
    for vertex in bn.V:
        bn.Vdata[vertex] = dict()
        bn.Vdata[vertex]["children"] = graphskeleton.getchildren(vertex)
        bn.Vdata[vertex]["parents"] = graphskeleton.getparents(vertex)

        # make placeholders for vals, cprob, and numoutcomes
        bn.Vdata[vertex]["vals"] = []
        if (bn.Vdata[vertex]["parents"] == []):
            bn.Vdata[vertex]["cprob"] = []
        else:
            bn.Vdata[vertex]["cprob"] = dict()

        bn.Vdata[vertex]["numoutcomes"] = 0

    #print '1 ------'

    # determine which outcomes are possible for each node
    for sample in data:
        for vertex in bn.V:
            if (sample[vertex] not in bn.Vdata[vertex]["vals"]):
                bn.Vdata[vertex]["vals"].append(sample[vertex])
                bn.Vdata[vertex]["numoutcomes"] += 1

    # lay out probability tables, and put a [num, denom] entry in all spots:

    # define helper function to recursively set up cprob table
    def addlevel(vertex, _dict, key, depth, totaldepth):
        if depth == totaldepth:
            _dict[str(key)] = []
            for _ in range(bn.Vdata[vertex]["numoutcomes"]):
                _dict[str(key)].append([0, 0])
            return
        else:
            for val in bn.Vdata[bn.Vdata[vertex]["parents"][depth]]["vals"]:
                ckey = key[:]
                ckey.append(str(val))
                addlevel(vertex, _dict, ckey, depth + 1, totaldepth)
    #print '2 ------'
    # put [0, 0] at each entry of cprob table
    for vertex in bn.V:
        if (bn.Vdata[vertex]["parents"]):
            root = bn.Vdata[vertex]["cprob"]
            numparents = len(bn.Vdata[vertex]["parents"])
            addlevel(vertex, root, [], 0, numparents)
        else:
            for _ in range(bn.Vdata[vertex]["numoutcomes"]):
                bn.Vdata[vertex]["cprob"].append([0, 0])
    #print '3 ------'
    # fill out entries with samples:
    for sample in data:
        for vertex in bn.V:
            # print 'vertex ', vertex

            # compute index of result
            rindex = bn.Vdata[vertex]["vals"].index(sample[vertex])
            # print 'rindex ', rindex

            # go to correct place in Vdata
            if bn.Vdata[vertex]["parents"]:
                pvals = [str(sample[t]) for t in bn.Vdata[vertex]["parents"]]
                lev = bn.Vdata[vertex]["cprob"][str(pvals)]
            else:
                lev = bn.Vdata[vertex]["cprob"]

            # increase all denominators for the current condition
            for entry in lev:
                entry[1] += 1

            # increase numerator for current outcome
            lev[rindex][0] += 1
            # print 'lev ', lev
    #print '4 ------'
    ########################### LAPLACE SMOOTHING TO AVOID ZERO DIVISION ERROR WHEN WE HAVE EMPTY BINS #############################
    #"""
    for vertex in bn.V:
        #print 'vertex ', vertex
        # print bn.V[vertex]
        numBins = bn.Vdata[vertex]['numoutcomes']

        if not (bn.Vdata[vertex]["parents"]):  # has no parents
        #    for i in range(len(bn.Vdata[vertex]['cprob'])):
        #        bn.Vdata[vertex]['cprob'][i][0] += 1  # numerator (count)
        #        bn.Vdata[vertex]['cprob'][i][1] += numBins  # denomenator (total count)

            for counts in bn.Vdata[vertex]['cprob']:
                counts[0] += 1  # numerator (count)
                counts[1] += numBins  # denomenator (total count)
        else:

            countdict = bn.Vdata[vertex]['cprob']

            for key in countdict.keys():
                for counts in countdict[key]:
                    counts[0]+=1
                    counts[1]+=numBins

            #print '5 ------'
            """
             # OPTIONAL: converts cprob from dict into df, does laplace smoothing, then (missing) maps back to dict
            bincounts = pd.DataFrame.from_dict(bn.Vdata[vertex]['cprob'], orient='index')
            #print bincounts

            for columnI in range (0, bincounts.shape[1]):
                for rowI in range (0,bincounts.shape[0]):
                    bincounts[columnI][rowI]=[bincounts[columnI][rowI][0]+1,bincounts[columnI][rowI][1]+numBins]
            #print bincounts
            """

    #print 'max def ', bn.Vdata['max_def']
    #print 'EAx ', bn.Vdata['EAx']
    #print 'EAy ', bn.Vdata['EAy']
    #print 'mass ', bn.Vdata['mass']
    #print 'cog_x ', bn.Vdata['cog_x']
    #print 'cog_z ', bn.Vdata['cog_z']

    #"""
    #print '6 ------'
    ######################################################################################

    # convert arrays to floats
    for vertex in bn.V:
        if not bn.Vdata[vertex]["parents"]:
            bn.Vdata[vertex]["cprob"] = [x[0] / float(x[1]) for x in bn.Vdata[vertex]["cprob"]]
        else:
            for key in bn.Vdata[vertex]["cprob"].keys():
                try:
                    bn.Vdata[vertex]["cprob"][key] = [x[0] / float(x[1]) for x in bn.Vdata[vertex]["cprob"][key]]

                # default to even distribution if no data points
                except ZeroDivisionError:

                    bn.Vdata[vertex]["cprob"][key] = [1 / float(bn.Vdata[vertex]["numoutcomes"]) for x in
                                                      bn.Vdata[vertex]["cprob"][key]]

    # return cprob table with estimated probability distributions
    return bn

def condprobve2(self, query, evidence):
    '''
    Eliminate all variables in *factorlist* except for the ones queried. Adjust all distributions for the evidence given. Return the probability distribution over a set of variables given by the keys of *query* given *evidence*. 

    Arguments:
        1. *query* -- A dict containing (key: value) pairs reflecting (variable: value) that represents what outcome to calculate the probability of. 
        2. *evidence* -- A dict containing (key: value) pairs reflecting (variable: value) that represents what is known about the system.

    Attributes modified:
        1. *factorlist* -- Modified to be one factor representing the probability distribution of the query variables given the evidence.

    The function returns *factorlist* after it has been modified as above.

    Usage example: this code would return the distribution over a queried node, given evidence::

        import json

        from libpgm.graphskeleton import GraphSkeleton
        from libpgm.nodedata import NodeData
        from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
        from libpgm.tablecpdfactorization import TableCPDFactorization

        # load nodedata and graphskeleton
        nd = NodeData()
        skel = GraphSkeleton()
        nd.load("../tests/unittestdict.txt")
        skel.load("../tests/unittestdict.txt")

        # toporder graph skeleton
        skel.toporder()

        # load evidence
        evidence = dict(Letter='weak')
        query = dict(Grade='A')

        # load bayesian network
        bn = DiscreteBayesianNetwork(skel, nd)

        # load factorization
        fn = TableCPDFactorization(bn)

        # calculate probability distribution
        result = fn.condprobve(query, evidence)

        # output
        print json.dumps(result.vals, indent=2)
        print json.dumps(result.scope, indent=2)
        print json.dumps(result.card, indent=2)
        print json.dumps(result.stride, indent=2)

    '''
    assert (isinstance(query, dict) and isinstance(evidence, dict)), "First and second args must be dicts."
    ## need to modify and add 1 to the zeros here but need frequency count

    #print 'factor list ', self.factorlist
    #print 'factor list0 len ', self.factorlist[0].vals
    #print 'factor list1 len ', self.factorlist[1].vals
    #print 'factor list2 len  ', self.factorlist[2].vals
    #print 'factor list3 len  ', self.factorlist[3].vals
    #print 'factor list4 len  ', self.factorlist[4].vals
    #print 'factor list5 len  ', self.factorlist[5].vals

    eliminate = self.bn.V[:]
    # print 'bn ', self.bn.V
    for key in query.keys():
        eliminate.remove(key)
    for key in evidence.keys():
        #print 'key ', key
        eliminate.remove(key)
    # print 'el bn ',eliminate
    # modify factors to account for E = e
    for key in evidence.keys():
        #print 'key ', key
        for x in range(len(self.factorlist)):
            if (self.factorlist[x].scope.count(key) > 0):
                #print self.factorlist[x].scope.count(key)
                self.factorlist[x].reducefactor(key, evidence[key])
                #print self.factorlist[x].vals
        for x in reversed(range(len(self.factorlist))):
            if (self.factorlist[x].scope == []):
                del (self.factorlist[x])

    # eliminate all necessary variables in the new factor set to produce result
    # print 'el bn ', eliminate
    # print 'factor list before elimination ', self.factorlist
    self.sumproductve(eliminate)
    # print 'el factor list', self.sumproductve(eliminate).factorlist
    #print 'factor list after elimination ', self.factorlist.vals
    # normalize result
    summ = 0.0
    lngth = len(self.factorlist.vals)
    for x in range(lngth):
        summ += self.factorlist.vals[x]
    #print 'summ ', summ
    for x in range(lngth):
        a = float(self.factorlist.vals[x])
        #print 'a ', a
        a = a / summ

    # return table
    # print 'self.stride ', self.stride
    return self.factorlist

"""
def predictquerieddistribution(queries, evidence, baynet): # TODO: extend to handle multiple query nodes

    posterior_distributions = []
    fn = TableCPDFactorization(baynet)

    for query in queries:
        # result = fn.condprobve(query, evidence) #from library
        result = condprobve2(fn, query, evidence) #written here
        probabilities = printdist(result, baynet)
        probabilities.sort_values(['max_def'], inplace=True)  # make sure probabilities are listed in order of bins
        posterior_distributions.append(probabilities)


    return posterior_distributions
"""

def inferPosteriorDistribution(queries, evidence, baynet):  # TODO: extend to handle multiple query nodes

    fn = TableCPDFactorization(baynet)

    # result = fn.condprobve(query, evidence) #from library
    result = condprobve2(fn, queries, evidence)  # written here
    print('result.vals '), result.vals
    probabilities = printdist(result, baynet)
    # for index,key in queries:
    probabilities.sort_values(['max_def'], inplace=True)  # make sure probabilities are listed in order of bins

    return probabilities

def laplacesmooth(bn): # TODO: update this function as per code written in condprobve or lmeestimateparams
    for vertex in bn.V:
        print('vertex '), vertex
        # print bn.V[vertex]
        numBins = bn.Vdata[vertex]['numoutcomes']

        if not (bn.Vdata[vertex]["parents"]):  # has no parents
            for i in range(len(bn.Vdata[vertex]['cprob'])):
                bn.Vdata[vertex]['cprob'][i][0] += 1  # numerator (count)
                bn.Vdata[vertex]['cprob'][i][1] += numBins  # denomenator (total count)
        else:
            for i in range(numBins):
                binindex = [str(float(i))]
                bincounts = bn.Vdata[vertex]['cprob'][str(binindex)]
                for j in range(len(bincounts)):
                    bincounts[j][0] += 1  # numerator (count)
                    bincounts[j][1] += numBins  # denomenator (total count)

    return bn

def buildBN(trainingData, binstyleDict, numbinsDict, **kwargs): # need to modify to accept skel or skelfile

    discretized_training_data, bin_ranges = discretizeTrainingData(trainingData, binstyleDict, numbinsDict, True)
    print('discret training '),discretized_training_data


    if 'skel'in kwargs:
        # load file into skeleton
        if isinstance(kwargs['skel'], basestring):
            skel = GraphSkeleton()
            skel.load(kwargs['skel'])
            skel.toporder()
        else:
            skel = kwargs['skel']

    # learn bayesian network
    learner = PGMLearner()
    # baynet = learner.discrete_mle_estimateparams(skel, discretized_training_data)
    # baynet = discrete_estimatebn(learner, discretized_training_data, skel, 0.05, 1)
    baynet = discrete_mle_estimateparams2(skel,discretized_training_data)  # using discrete_mle_estimateparams2 written as function in this file, not calling from libpgm

    return baynet

def expectedValue (binRanges, probabilities):

    expectedV = 0.0

    for index, binrange in enumerate(binRanges):

        v_max = binrange[0]
        v_min = binrange[1]

        meanBinvalue = ((v_max - v_min) / 2) + v_min

        expectedV += meanBinvalue * probabilities[index]

    return expectedV

def discretize (dataframe, binRangesDict, plot=False):

    binnedDf = pd.DataFrame().reindex_like(dataframe)

    binCountsDict = copy.deepcopy(binRangesDict)  # copy trainingDfDiscterizedRangesDict
    for key in binCountsDict:
        for bin in binCountsDict[key]:
            del bin[:]
            bin.append(0)

    for varName in binRangesDict.keys():
        # load discretized ranges belonging to varName in order to bin in
        discreteRanges = binRangesDict.get(varName)
        #print discreteRanges
        # binCounts = binCountsDict[varName]

        index = 0
        # for item1, item2 in trainingDf[varName], valBinnedDf[varName]:
        for item1 in dataframe[varName]:

            for i in range(len(discreteRanges)):
                binRange = discreteRanges[i]

                ############ bin training data #############
                #print 'bin range is ', binRange
                #print 'value to bin ', item1

                if i==0: # #if this is first bin then bin numbers larger or equal than min num and less or equal than max num (basically, include min num)
                    if binRange[0] <= item1 <= binRange[1]:
                        #print item1,' is binned within ',binRange
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                else: #if not first bin bin numbers less or equal to max num
                    if binRange[0] < item1 <= binRange[1]:
                        #print item1,' is binned within ',binRange
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                # catch values outside of range (smaller than min)
                if i == 0 and binRange[0] > item1:
                    #print 'the value ', item1, 'is smaller than the minimum bin', binRange[0]
                    binnedDf.iloc[index][varName] = i
                    binCountsDict[varName][i][0] += 1

                # catch values outside of range (larger than max)
                if i == len(discreteRanges) - 1 and binRange[1] < item1:
                    #print 'the value ', item1, 'is larger than the maximum bin', binRange[1]
                    binnedDf.iloc[index][varName] = i
                    binCountsDict[varName][i][0] += 1

            index += 1

    binnedData = binnedDf.to_dict(orient='records') # a list of dictionaries

    return binnedData, binnedDf, binCountsDict

"""
def getBinRangesAuto (dataFrame, targets):

    transformer = MDLP(min_depth=3) # set min_depth to 2 avoid empty cut points

    ### get bin ranges for inputs

    varNames = list(dataFrame.columns.values)
    for target in targets:
        if target in varNames: varNames.remove(target)

    y = dataFrame[targets[0]].tolist()
    X = dataFrame[varNames].values.tolist()

    transformer.fit_transform(X, y)

    allcutpts = transformer.cut_points_

    DiscterizedRangesDict = {}

    for variable in allcutpts:
        binranges = []
        for i in range(1, len(allcutpts[variable])):
            cutpt = allcutpts[variable][i - 1]
            nextcutpt = allcutpts[variable][i]
            binranges.append([cutpt, nextcutpt])

        DiscterizedRangesDict[varNames[variable]] = binranges

    ### get bin ranges for target/s
    for target in targets:
        # numbins = numBinsDict.get(target)
        targetbinranges = bins(max(dataFrame[target]), min(dataFrame[target]), 6)

        DiscterizedRangesDict[target] = targetbinranges

    for item in DiscterizedRangesDict: print item, ' -- ', DiscterizedRangesDict[item]

    return DiscterizedRangesDict

def mdlp (dataFrame, targets):
    ### get bin ranges for single input

    transformer = MDLP(min_depth=3)  # set min_depth to 2 avoid empty cut points

    varNames = list(dataFrame.columns.values)
    for target in targets:
        if target in varNames: varNames.remove(target)

    y = dataFrame[targets[0]].tolist()
    X = dataFrame[varNames].values.tolist()

    transformer.fit_transform(X, y)

    allcutpts = transformer.cut_points_

    DiscterizedRangesDict = {}


    binranges = []
    for i in range(1, len(allcutpts[variable])):
        cutpt = allcutpts[variable][i - 1]
        nextcutpt = allcutpts[variable][i]
        binranges.append([cutpt, nextcutpt])

        DiscterizedRangesDict[varNames[variable]] = binranges
"""
def getBinRanges (dataframe, binTypeDict, numBinsDict):

    # numBinDict should be in the form of {max_def: 10, moment_inertia: 5, ...}

    #trainingDf = pd.DataFrame(self.unbinnedDF)
    #trainingDf.columns = trainingDf.iloc[0]
    #trainingDf = trainingDf[1:]
    #print trainingDf

    trainingDfDiscterizedRanges = []
    trainingDfDiscterizedRangesDict = {}

    # loop through variables in trainingDf (columns) to discretize into ranges according to trainingDf

    #TODO: the names should be taken from an origina list of BN nodes not assuming all vars in df. THis will allow to use one dtaframe, such that we can
    #TODO: ... build a bn on any of the columns selected from the csv file.
    #for varName in list(dataframe):
    for varName in binTypeDict.keys():
        # key = traininDf.columns
        # if 'p', discretise variable i, using percentiles, if 'e', discretise using equal bins
        if binTypeDict[varName] == 'p':
            trainingDfDiscterizedRanges.append(percentile_bins(dataframe[varName], numBinsDict.get(varName)))  # adds to a list
            trainingDfDiscterizedRangesDict[varName] = percentile_bins(dataframe[varName], numBinsDict.get(varName))  # adds to a dictionary
        elif 'e':
            trainingDfDiscterizedRanges.append(bins(max(dataframe[varName]), min(dataframe[varName]),numBinsDict.get(varName)))  # adds to a list
            trainingDfDiscterizedRangesDict[varName] = bins(max(dataframe[varName]), min(dataframe[varName]),numBinsDict.get(varName))  # adds to a dictionary

        #elif 'a':
        #    trainingDfDiscterizedRangesDict[varName] =

        # TODO: add other option: elif 'auto(mlp)':

    return trainingDfDiscterizedRangesDict

def generateErrors (predictedTargetPosteriors, testingData, binnedTestingData, binRanges, target):

    posteriorPDmeans = []



    for posterior in predictedTargetPosteriors:

        posteriorPDmeans.append(expectedValue((binRanges[target]), posterior))

    mse = mean_squared_error(testingData[target], posteriorPDmeans)
    # mse = mean_squared_error(unbinnedTargetActual[targetList[0]], posteriorPDmeans)
    rmse = math.sqrt(mse)

    #print 'binnedTestingData[target] ', binnedTestingData[target]
    #print 'predictedTargetPosteiors ', predictedTargetPosteriors

    loglossfunction = sklearn.metrics.log_loss(binnedTestingData[target], predictedTargetPosteriors,normalize=True, labels=range(0, len(binRanges[target])))
    norm_distance_errors = distribution_distance_error(binnedTestingData[target], predictedTargetPosteriors,testingData[target], binRanges[target], False)

    correct_bin_probabilities = []
    for p in range(len(testingData[target])):
        correct_bin_probabilities.append(predictedTargetPosteriors[p][binnedTestingData[target][p]])


    return float(rmse),float(loglossfunction),norm_distance_errors,correct_bin_probabilities

"""
def predictquerieddistribution(queries, evidence, baynet): # TODO: extend to handle multiple query nodes

    posterior_distributions = []
    fn = TableCPDFactorization(baynet)

    for query in queries:
        # result = fn.condprobve(query, evidence) #from library
        result = condprobve2(fn, query, evidence) #written here
        probabilities = printdist(result, baynet)
        probabilities.sort_values(['max_def'], inplace=True)  # make sure probabilities are listed in order of bins
        posterior_distributions.append(probabilities)


    return posterior_distributions
"""

def BNskelFromCSV (csvdata, targets):

    #TODO: must know how to swap direction of too many inputs into a node

    ######## EXTRACT HEADER STRINGS FROM CSV FILE ########
    skel = GraphSkeleton()
    BNstructure = {}
    inputVerts = []


    # if data is a filepath
    if isinstance(csvdata, basestring):

        dataset = []

        with open(csvdata, 'rb') as csvfile:
            lines = csv.reader(csvfile)

            for row in lines:
                dataset.append(row)

        allVertices = dataset [0]

    else: allVertices = csvdata[0]

    BNstructure ['V'] = allVertices
    skel.V = allVertices


    for verts in allVertices:
        if verts not in targets:
            inputVerts.append(verts)

    #target, each input
    edges = []

    if len(inputVerts) > len(targets):
        for target in targets:

            for input in inputVerts:
                edge = [target, input]
                edges.append(edge)

        BNstructure ['E'] = edges
        skel.E = edges

    else:
        for input in inputVerts:
            for target in targets:
                edge = [input, target]
                edges.append(edge)
                
        BNstructure['E'] = edges
        skel.E = edges


    skel.toporder()

    return skel

