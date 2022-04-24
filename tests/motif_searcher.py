# here I will do some motifs searching, with multiprocessing to speed it up

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
import pathlib
import glob

import json

import networkx as nx
import random

import timeit
from multiprocessing import Process, Queue, Pool


#define functions
def generateMotifsX():
    """
    Generates motifs using networkx
    """
    
    
    motifs = [];
    #3 nodes (2)
    motifs.append(['M31',nx.Graph()]); motifs.append(['M32',nx.Graph()])

    motifs[0][1].add_edges_from([[0,1], [1,2]]);        #open triangle
    motifs[1][1].add_edges_from([[0,1], [1,2], [2,0]]); #closed triangle
    #4 nodes (6)
    motifs.append(['star', nx.Graph()]);
    #motifs.append(['chain', nx.Graph()]); 
    #motifs.append(['winged', nx.Graph()]);
    motifs.append(['box', nx.Graph()]);
    motifs.append(['s-clique', nx.Graph()]);
    motifs.append(['clique', nx.Graph()]);

    motifs[2][1].add_edges_from([[0,1], [0,2], [0,3]]);                      #star
    #motifs[3][1].add_edges_from([[0,1], [1,2], [2,3]]);                      #chain
    #motifs[4][1].add_edges_from([[0,1], [0,2], [0,3], [2,3]]);               #triangle with one wing
    motifs[3][1].add_edges_from([[0,1], [1,2], [2,3], [3,0]]);               #box
    motifs[4][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [0,2]]);        #semi-clique
    motifs[5][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [0,2], [1,3]]); #clique

    """
    #5 nodes (21)
    motifs.append(['M501', nx.Graph()]); motifs.append(['M502', nx.Graph()]); motifs.append(['M503', nx.Graph()])
    motifs.append(['M504', nx.Graph()]); motifs.append(['M505', nx.Graph()]); motifs.append(['M506', nx.Graph()])
    motifs.append(['M507', nx.Graph()]); motifs.append(['M508', nx.Graph()]); motifs.append(['M509', nx.Graph()])
    motifs.append(['M510', nx.Graph()]); motifs.append(['M511', nx.Graph()]); motifs.append(['M512', nx.Graph()])
    motifs.append(['M513', nx.Graph()]); motifs.append(['M514', nx.Graph()]); motifs.append(['M515', nx.Graph()])
    motifs.append(['M516', nx.Graph()]); motifs.append(['M517', nx.Graph()]); motifs.append(['M518', nx.Graph()])
    motifs.append(['M519', nx.Graph()]); motifs.append(['M520', nx.Graph()]); motifs.append(['M521', nx.Graph()])

    motifs[8][1].add_edges_from([[0,1], [1,2], [2,3], [3,4]]);   # chain
    motifs[9][1].add_edges_from([[0,1], [1,2], [2,3], [2,4]]);   # split
    motifs[10][1].add_edges_from([[0,1], [0,2], [0,3], [0,4]]);  # cross
    motifs[11][1].add_edges_from([[0,1], [0,2], [1,2], [1,3], [2,4]]); # triangle with 2 wings
    motifs[12][1].add_edges_from([[0,1], [0,2], [1,2], [0,3], [3,4]]); # triangle with 1 long chain
    motifs[13][1].add_edges_from([[0,1], [0,2], [1,2], [0,3], [0,4]]); # triangle with one node 2 wings
    motifs[14][1].add_edges_from([[0,1], [1,2], [2,3], [3,4], [4,0]]); # penta-box
    motifs[15][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [4,0]]); # box with chain
    motifs[16][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [4,0], [0,2]]);  # verti crossed semi clique 1 wing
    motifs[17][1].add_edges_from([[0,1], [1,2], [2,0], [0,3], [3,4], [4,0]]);  # joiend 2 triangle
    motifs[18][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [4,0], [1,3]]);  # horti crossed semi clique 1 wing
    motifs[19][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [0,4], [4,2]]);  # filled quadrangle
    motifs[20][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [0,4], [4,1]]);  # house
    motifs[21][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [0,2], [0,4], [4,2]]);  # strange diamond
    motifs[22][1].add_edges_from([[0,1], [1,2], [2,0], [0,3], [1,3], [2,3], [0,4]]);
    motifs[23][1].add_edges_from([[0,1], [1,2], [2,0], [0,3], [1,3], [0,4], [3,4]]); # trapeze
    motifs[24][1].add_edges_from([[0,1], [1,2], [2,0], [0,3], [1,3], [3,4], [2,4]]);
    motifs[25][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [0,2], [0,4], [4,2], [1,4]]);
    motifs[26][1].add_edges_from([[0,1], [1,2], [2,3], [3,0], [0,4], [1,4], [2,4], [3,4]]);
    motifs[28][1].add_edges_from([[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4]]); #almost full penta
    motifs[28][1].add_edges_from([[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]);  #full penta
    """
    
    return motifs

def calcRevelancy(graph,motifs_gen=generateMotifsX, seed=726, random_num=5, reroll=0.3):
    """
    Gives the revelanvy of given motifs (Z SCORE)
    (m_obs - m_rand_avg) / (sigma_rand)
    
    INPUT:
        graoh: graph object for comparison
        motif_gen: func, generates motifs
        seed= integer for seeding random choosings
        random_num = amount of random networks to be generated
        reroll= float, (0,1], how much percentage of the links should be remade
    OUTPUT:
        motifs_order: name of the motfis, motif_gen gives it
        motifs_match_count: array for motifs found in graph, random generated graph and revelancy
    """
    #generate random graph
    G_a = graph
    #generate random_num random graphs (sufficient for my needs)
    randoms = []
    for i in range(random_num):
        G_random_base = nx.Graph()
        G_random_base.add_edges_from(G_a.edges)
        print("Generating randomized network number {}".format(i))
        randoms.append(nx.double_edge_swap(G_random_base,
                                           int(reroll*len(list(G_random_base.edges))),
                                           200*int(reroll*len(list(G_random_base.edges))),
                                           seed-i))
    
    #containers
    motifs_order0 = []
    motifs_match_count0 = []
    
    motifs = motifs_gen()
    
    #let's have the subgraph matchers
    for i in motifs:
        
        GM_a_i = 0
        GM_r_i = np.zeros((random_num))
        
        cnt_a_i = 0
        
        
        sys.stdout.write("\rAt {}, {}/{}\t\t".format(i[0],1,random_num+1))
        sys.stdout.flush()
        
        #get the graph matcher
        GM_a_i = nx.algorithms.isomorphism.GraphMatcher(G_a,i[1])
        #count how many matches it has with iter
        for iso in GM_a_i.subgraph_isomorphisms_iter():
            cnt_a_i+=1
        
        #let's do link randomization from the original network
        for j in range(random_num):
            cnt_r_i = 0
            sys.stdout.write("\rAt {}, {}/{}\t\t".format(i[0],j+1+1,random_num+1))
            sys.stdout.flush()
            GM_r_iso = nx.algorithms.isomorphism.GraphMatcher(randoms[j],i[1])
            #count how many matches it has with iter
            for iso in GM_r_iso.subgraph_isomorphisms_iter():
                cnt_r_i+=1
            GM_r_i[j] = cnt_r_i

        #now we should have the observed network and the random networks random app
        


        motifs_order0.append([i[0]])
        revelancy = 0
        if not (np.var(GM_r_i) == 0):
            revelancy = (cnt_a_i-np.mean(GM_r_i))/(np.var(GM_r_i)**(1/2))

        motifs_match_count0.append([cnt_a_i, np.mean(GM_r_i), revelancy])
    
    print("\nFinished")
    
    return np.array(motifs_order0), np.array(motifs_match_count0)

def generateRandomizedNetworks(graph,seed,reroll):
    G_random_base = nx.Graph()
    G_random_base.add_edges_from(graph.edges)
    G_random_base = nx.double_edge_swap(G_random_base,
                                        int(reroll*len(list(G_random_base.edges))),
                                        200*int(reroll*len(list(G_random_base.edges))),
                                        seed)
    
    return G_random_base

def countIsoIters(graph,motif):
    
    cnt_r_i = 0

    GM_r_iso = nx.algorithms.isomorphism.GraphMatcher(graph,motif)
    #count how many matches it has with iter
    for iso in GM_r_iso.subgraph_isomorphisms_iter():
        cnt_r_i+=1
    
    return cnt_r_i

def calcRevelancyParallel(graph,motifs_gen=generateMotifsX, seed=726, random_num=5, reroll=0.3):
    """
    Gives the revelanvy of given motifs (Z SCORE)
    (m_obs - m_rand_avg) / (sigma_rand)
    Creates a Pool to draw workers from... (multiprocessing)
    
    INPUT:
        graoh: graph object for comparison
        motif_gen: func, generates motifs
        seed= integer for seeding random choosings
        random_num = amount of random networks to be generated
        reroll= float, (0,1], how much percentage of the links should be remade
    OUTPUT:
        motifs_order: name of the motfis, motif_gen gives it
        motifs_match_count: array for motifs found in graph, random generated graph and revelancy
    """
    #get the workers from the pool
    pool = Pool(random_num+1)    
    
    #generate random graph
    G_a = graph
    #generate random_num random graphs (sufficient for my needs)
    #parallelize the randomization
    randoms = pool.starmap(generateRandomizedNetworks,
                           [(G_a,seed-i,reroll) for i in range(random_num)])
    #have to combine them together
    graphs = list(randoms)
    graphs.append(G_a)
    
    #containers
    motifs_order0 = []
    motifs_match_count0 = []
    
    motifs = motifs_gen()
    #print(motifs)
        
    #let's have the subgraph matchers
    for i in motifs:       
        funs = [(graphs[j], i[1]) for j in range(len(graphs))]
        #print(funs)
        
        sys.stdout.write("\rAt {}\t\t".format(i[0]))
        sys.stdout.flush()
        
        motifs_in_graphs = pool.starmap(countIsoIters, funs)
        
        motifs_order0.append([i[0]])
        revelancy = 0
        if not (np.var(motifs_in_graphs[:-1]) == 0):
            revelancy = (motifs_in_graphs[-1] - np.mean(motifs_in_graphs[:-1]))/(np.var(motifs_in_graphs[:-1])**(1/2))

        motifs_match_count0.append([motifs_in_graphs[-1], np.mean(motifs_in_graphs[:-1]), revelancy])
    
    print("\nFinished")
    
    return np.array(motifs_order0), np.array(motifs_match_count0)


#main to run main
if __name__ == "__main__":
    
    #load in the file
    print("Loading")
    #multi_sources_undirected_linklist
    with open("../SciMod/partial_results/multi_sources_undirected_linklist.txt", "rb") as f:
        second_comp_links = np.loadtxt(f)
        
    print("Loading finished")
    
    
    G = nx.Graph()                      #create graph
    G.add_edges_from(second_comp_links) #add egdes
    

    print(len(list(G.nodes)), len(list(G.edges) ) )
    """
    start = timeit.default_timer()
    m_o, m_m_c = calcRevelancy(G)
    stop = timeit.default_timer()
    """
    
    start2 = timeit.default_timer()
    m_o2, m_m_c2 = calcRevelancyParallel(G)
    stop2 = timeit.default_timer()
    
    print(#"\nTime in single-threaded (netx): " + str(stop - start) +
          "\nTime in multiprocessing (netx): " + str(stop2 - start2) #+
          #"\nAchieved speedup: " + str((stop-start) / (stop2-start2))
         ) 
    
    with open("../Scimod/partial_results/motif_finding_results.npy", "wb") as f:
        second_comp_links = np.save(f, m_m_c2)
    print("Finished running!")