# Eliza Kraule
# emk6
# COMP 182 Spring 2021 - Homework 4, Problem 3

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from comp182.py, provided.py, and autograder.py,
# but they have to be copied over here.

from collections import *


import numpy
import matplotlib.pyplot as plt
import pylab
import types
import time
import math
import copy
import random


# all the functions copied from comp182 and provided and autograder:
def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g
def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))
def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g
def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.

    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.

    Arguments:
    num_nodes -- The number of nodes in the returned graph.

    Returns:
    A complete graph in dictionary form.
    """
    result = {}

    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value:
                result[node_key].add(node_value)

    return result
def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns:
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result
def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()

    ### Iterate through each possible edge and add it with
    ### probability p.
    for u in range(n):
        for v in range(u+1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

    return g
def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)
def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments:
    data     -- a list of dictionaries, each of which will be plotted
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data) - len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)
def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)
def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals
def _plot_dict_bar(d, xmin=None, label=None):
    """
    Plot data in the dictionary d on the current plot as bars.

    Arguments:
    d     -- dictionary
    xmin  -- optional minimum value for x axis
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if xmin == None:
        xmin = min(xvals) - 1
    else:
        xmin = min(xmin, min(xvals) - 1)
    if label:
        pylab.bar(xvals, yvals, align='center', label=label)
        pylab.xlim([xmin, max(xvals) + 1])
    else:
        pylab.bar(xvals, yvals, align='center')
        pylab.xlim([xmin, max(xvals) + 1])



#functions I have written:

def compute_largest_cc_size(g):
    """
    returns the size of largest connected component (the number of nodes in the component) in a graph

    Arguments:
    g        -- a graph

    Returns:
    CCsize -- the size of the connected component
    """

    count = 1  # initialize with one to count the initial node
    CCsize = 0 # initialize the CCsize as 0
    Q = deque()  ##intialize queue
    A = list(g.keys())  # make a copy of all the keys in dictionary
    for i in A: #check each node
        Q.append(i) #put it in que
        A.remove(i) #remove a node that has been checked from the list
        while len(Q) > 0: #while the que is not empty
            j = Q.pop()
            for neighbor in g.get(j): #check all the neighbours of node j
                if neighbor in A:
                    Q.append(neighbor)
                    A.remove(neighbor)
                    count = 1 + count #count the neighbouring nodes
        if count > CCsize: #check if the size of this component is larger than the previous
            CCsize = count
            count = 1
    return CCsize
def random_attack(g,percentage):
    """ RANDOM ATTACK
    this function removes a certain percentage of nodes randomly
    from graph and returns a list of dictionaries where the key is
     number of nodes removed and the value is the size of largest connected component

    Arguments:
    g        -- a graph
    percentage -- percentage of nodes to be removed

    Returns:
    list_g -- list of dictionaries where key is the number of nodes removed and value is size of largest connected component in the respective graph
    """
    nodes_removed=0
    nodes = len(g)
    CC=compute_largest_cc_size(g)
    dict_g = dict({0:CC})
    rem_nodes = round(percentage * nodes)
    for n in range(rem_nodes):
        i = list(g.keys())
        r = random.choice(i)
        g.pop(r)
        #nodes = len(g)
        for v in g.values():
            if (r) in v:
                v.remove(r)
        nodes_removed = nodes_removed + 1
        CC = compute_largest_cc_size(g)
        dict_g[nodes_removed]= CC
    dict_g_copy=dict_g.copy()
    list_g=[]
    list_g.append(dict_g)
    return list_g
def target_attack(g,percentage):
    """"TARGET ATTACK
    this function finds the degrees of nodes and removes a certain percentage of them
     in the sequence of largest to smallest degree

    Arguments:
    g        -- a graph
    percentage -- percentage of nodes to be removed

    Returns:
    list_target -- list of dictionaries where key is the number of nodes removed and value is size of
     largest connected component in the resulting graph
    """
    nodes_removed=0
    nodes = len(list(g.keys())) # number of nodes
    CC = compute_largest_cc_size(g) #largest connected component
    dict_target = dict({0:CC})
    rem_nodes = round(percentage * nodes)
    for i in range(rem_nodes):
        nodes_list = list(g.keys())
        C = (list(map(len, g.values())))  # finds the  degree of each node
        A = C.index(max(C))# find the index of the node of the largest degree
        nod=nodes_list[A]
        g.pop(nod)  # remove the node of largest degree
        for v in g.values():  # removes the edges of the node of largest degree
            if nod in v:
                v.remove(nod)
        nodes_removed = nodes_removed + 1
        CComp = compute_largest_cc_size(g)
        dict_target[nodes_removed] = CComp
        dict_target_copy = dict_target.copy()
    list_target = []
    list_target.append(dict_target)
    return list_target


#tests for functions:

#for testing i will use the two following graphs:
g2_test = {0:{2,4,5 ,8}, 1:{5}, 2:{0,4,6}, 3:{7,10}, 4:{6,2,0}, 5:{11,1,0}, 6:{4,2} ,7:{3,10} ,8:{0}, 9:{}, 10:{3,7}, 11:{5}}
g_test={0:{}}
#g2_test and g_test are small graphs used to test and debug functions as i can draw out  each of them and check if the code returns the correct values
# some characteristics of the g2_graph are that it consists of 12 nodes and the largest connected component is 8
#g_test is a graph of a single node

""""test compute_largest_cc_size"""

""""
tests are run by using graphs g_test and g2_test which are small graphs that i have made and 
can check the connected component size
1) test with g_test, this must return 1
2) test with g2_test, this must return 8
"""
g2_test = {0:{2,4,5 ,8}, 1:{5}, 2:{0,4,6}, 3:{7,10}, 4:{6,2,0}, 5:{11,1,0}, 6:{4,2} ,7:{3,10} ,8:{0}, 9:{}, 10:{3,7}, 11:{5}}
g_test={0:{}}

test1=compute_largest_cc_size(g_test)
if test1==1:
    print('test 1 for largest cc size passed')
else:
    print('test 1 largest cc size failed')
test2=compute_largest_cc_size(g2_test)
if test2==8:
    print('test 2 largest cc size passed')
else:
    print('test 2 largest cc size failed')


""""test random_attack

to test this function i will remove 1 node in each graph which is 100% of nodes in g_test and 10% of nodes in g2_test
the expected value of this function is a list of a dictionary where keys are the number of nodes removed and values 
are the size of connected components (including the intial graph were no nodes are removed) so to test it i will check 
the there exists a key in the dictionary which equals the number of nodes removed


1) test with g_test, this must return a list of a graph dictionary which has the key 1
2) test with g2_test, this must return a list of a graph dictionary which has the key 1
"""
g2_test = {0:{2,4,5 ,8}, 1:{5}, 2:{0,4,6}, 3:{7,10}, 4:{6,2,0}, 5:{11,1,0}, 6:{4,2} ,7:{3,10} ,8:{0}, 9:{}, 10:{3,7}, 11:{5}}
g_test={0:{}}

test1=random_attack(g_test,1)
if 1 in test1[0]:
    print('test 1  for random attack passed')
else:
    print('test 1 for random attack failed')

test2=random_attack(g2_test,0.1)
if 1 in test2[0]:
    print('test 2 for random attack passed')
else:
    print('test 2 for random attack failed')

"""test target_attack

to test this function i will remove 1 node in each graph which is 100% of nodes in g_test and 10% of nodes in g2_test
the expected value of this function is a list of a dictionary where keys are the number of nodes removed and values 
are the size of connected components (including the initial graph were no nodes are removed) so to test it i will check 
the there exists a key in the dictionary which equals the number of nodes removed. in addition to that i will check the 
size of the connected component when one node was removed as i can check this by drawing it.


1) test with g_test,  this must return a list of a graph dictionary which has the key 1 and the value (size of the connected component) is 0
2) test with g2_test, this must return a list of a graph dictionary which has the key 1 and the value(size of the connected component) is 5
"""
g2_test = {0:{2,4,5 ,8}, 1:{5}, 2:{0,4,6}, 3:{7,10}, 4:{6,2,0}, 5:{11,1,0}, 6:{4,2} ,7:{3,10} ,8:{0}, 9:{}, 10:{3,7}, 11:{5}}
g_test={0:{}}

test1=target_attack(g_test,1)
if 1 in test1[0] and test1[0].get(1)==0:
    print('test 1  for target attack passed')
else:
    print('test 1 for target attack failed')

test2=target_attack(g2_test,0.1)
if 1 in test2[0] and test2[0].get(1)==5:
    print('test 2 for target attack passed')
else:
    print('test 2 for target attack failed')

"""""tests finished"""

#build the upa and erdos_renyi graphs

#upa takes inputs:
#   n - num of nodes
#   m - num of edges per node

# number of nodes were determined by checking the lenght of the given graph
graph=read_graph("rf7.repr")
nodes_upa=len(graph) # returns the number 1347

# n=1347

# num of edges per node is the average degree in the graph which is total degree/num_nodes
degree=total_degree(graph) #returns 6224

#the number of edges in a graph are num_edges= sum_of_all_degrees/2
#therefore to find edges per node divide the number of edges by the number of nodes
#m=(6224/1347)=4.6 therefore we will use the value 5 for the input

#m=5

#erdos_renyi takes inputs:

# n - num of nodes
# p - probability of an edge
# the number of edges in a graph is half of that so build graphs with 6224/2

## the edge probability p= E/(n(n-1)/2) where E is total number of edges and
# n is total number of nodes and n(n-1)/2 is the number of edges in a complete graph.

# number of edges in a graph are total_degree/2 which is 3112 for the given graph
# plugging in the values the probability p=0.00343287


# calling the functions and plotting the results

#rf7 graph

#copy graphs
graph1=read_graph("rf7.repr")
graph2=read_graph("rf7.repr")

#call attack functons
list_rf7=target_attack(graph1,0.2)
random_rf7=random_attack(graph2,0.2)
#combine lists
list_rf7.extend(random_rf7)
#plot
plot_lines(list_rf7,'Graph rf7: Size of Connected Component Depending on Nodes Removed', 'Nodes Removed'
                ,'Size of Largest Connected Component (nodes)',['Target Attack','Random Attack'])

#upa graph

graph_upa=upa(1347,5)
graph_upa_copy=copy_graph(graph_upa)

list_upa=target_attack(graph_upa,0.2)
random_upa=random_attack(graph_upa_copy,0.2)

list_upa.extend(random_upa)

plot_lines(list_upa,'UPA: Size of Connected Component Depending on Nodes Removed', 'Nodes Removed'
           ,'Size of Largest Connected Component (nodes)',['Target Attack','Random Attack'])

#erdos renyi graph

graph_er=erdos_renyi(1347, 0.00343287)
graph_er_copy=copy_graph(graph_er)

list_er=target_attack(graph_er,0.2)
random_er=random_attack(graph_er_copy,0.2)

list_er.extend(random_er)

plot_lines(list_er,"Erdos Renyi:largest Connected Component Depending on Nodes Removed", 'Nodes Removed',
           'Size of Largest Connected Component (nodes)', ['Target Attack', 'Random Attack'])

