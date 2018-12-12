#!/usr/bin/env python3
# coding: utf-8

# ## To Do:
# 
# #### Inputs
# - boolean oligo
# - matrix_percents.txt file or .relabund file
# 
# #### Read in File
# - Function to convert matrix_percents.txt (direct oligotyping output) to RMN dataframe format
# - Function to convert .relabund (direct mothur output) to RMN dataframe format
# - Get all sets of pairs and triplets
# 
# #### Model
# - Piecewise functions and model function already done

# In[16]:


import pandas as pd
import numpy as np
import io
from scipy.stats import variation
from itertools import permutations
import math
import networkx as nx
import matplotlib.pyplot as plt
import sys


# In[3]:


def format_file(file, oligo = False):
    print("start format_file")
    if(oligo in ["true", "True", "T", "TRUE", "1"]):
        df = format_oligo(file)
    else:
        df = pd.read_table(file, sep = "\t")
        df = df.iloc[:,3:].transpose()
        df.insert(0, "OTU", list(df.index))
        df.index = range(0, len(df.index))
        df.iloc[:,1:] = df.iloc[:,1:]/100
    print("end format_file")
    return df

# In[4]:


def format_oligo(file):
    print("start format_oligo")
    df = pd.read_table(file, sep = "\t")
    if("numOtus" in df.columns):
        df = df.drop(["numOtus"], axis = 1)
    df = df.iloc[:,1:].transpose()
    df.insert(0, "Oligotype", list(df.index))
    df.index = range(0, len(df.index))
    df.iloc[:,1:] = df.iloc[:,1:]/100
    print("end format_oligo")
    return df

# In[7]:


def pairs(df):
    p = pd.DataFrame(list(permutations(list(df.index), r=2)))
    p = p.rename({0: "Om", 1: "Oc"}, axis = "columns")
    return p
def triplets(df):
    t = pd.DataFrame(list(permutations(list(df.index), r=2)))
    t = t.rename({0: "Om", 1: "Oc"}, axis = "columns")
    return t


# In[8]:


def high_comp(m, c):
    return 2*math.tanh(1.1*m)*(1-math.tanh(1.1*c))
def low_comp(m, c):
    return 1-(2*(1-math.tanh(1.1*m))*math.tanh(1.1*c))
def other(m, c):
    return math.tanh(1.1*m)-math.tanh(1.1*c)+0.5


# In[9]:


def SRP_model(m, c):
    if( (m < 0.5) & (c > 0.5) ): return high_comp(m, c)
    elif( (m > 0.5) & (c < 0.5) ): return low_comp(m, c)
    else: return other(m, c)


# In[11]:


def predict_SRP(df):
    print("start predict_SRP")
    p = pairs(df)
    exp_SRP_rows = list()
    for row in range(0, len(p.index)):
        m_index = p.iloc[row, 0]
        c_index = p.iloc[row, 1]
        m_list = df.iloc[m_index, 1:]
        c_list = df.iloc[c_index, 1:]
        exp_SRP = list()
        for i in range(0, len(m_list)):
            exp_SRP.append(SRP_model(m_list[i], c_list[i]))
        for i in range(0, len(df.index)-2):
            exp_SRP_rows.append(exp_SRP)
    exp_SRP_df = pd.DataFrame(exp_SRP_rows)
    exp_SRP_df.columns = list(range(1,len(exp_SRP_df.columns)+1))
    print("end predict_SRP")
    return exp_SRP_df


# In[12]:


def lack_of_fit(df, triplet_df, skip = -1):
    print("start lack_of_fit")
    exp = predict_SRP(df)
    obs = obs_SRP_df = df.iloc[list(triplet_df["Ot"]),1:].reset_index(drop=True)
    
    if(skip != -1):
        df = df.drop(df.columns[skip], axis=1)
        obs = obs.drop(obs.columns[skip-2], axis=1)
        exp = exp.drop(exp.columns[skip-2], axis=1)
        
    errors = list()
    for i in range(0,len(exp.columns)):
        errors.append(list(obs.iloc[:,i] - exp.iloc[:,i]))
    sse_df = pd.DataFrame(errors).transpose()**2
    sse_df.columns = list(range(1,len(sse_df.columns)+1))
    
    avg_SRP = df.mean(axis=1)
    sqdev_df = obs.subtract(list(avg_SRP[list(triplet_df["Ot"])]), axis=0)**2

    ssd = np.array(list(sqdev_df.sum(axis = 1)))
    sse = np.array(list(sse_df.sum(axis = 1)))
    print("end lack_of_fit")
    return list(sse/ssd)


# In[14]:


def adjust(df, triplet_df, lof):
    print("start adjust")
    lofn = list()
    for i in range(0, len(df.columns)):
        lofn.append(lack_of_fit(df, triplet_df, skip = i))
    lofn_df = pd.DataFrame(lofn).transpose()
    lofn_lof = lofn_df.divide(lof, axis=0)
    adj_df = lofn_lof.apply(np.log10, axis = 0).multiply(lofn_lof).abs()
    adj = adj_df.sum(axis = 1)
    print("end adjust")
    return adj


# In[15]:


def test_triplets(df, triplet_df, lof):
    print("start test_triplets")
    triplet_df["LoF"] = lof
    triplet_df["Ld"] = triplet_df["LoF"]/(1 - triplet_df["LoF"])
    adj = adjust(df, triplet_df, lof)
    triplet_df["Adj"] = adj
    triplet_df["I"] = lof*(1+adj)
    triplet_df["Om"] = list(df.iloc[triplet_df["Om"], 0])
    triplet_df["Oc"] = list(df.iloc[triplet_df["Oc"], 0])
    triplet_df["Ot"] = list(df.iloc[triplet_df["Ot"], 0])
    print("end test_triplets")
    return triplet_df


# In[19]:


def find_network(triplet_df, L):
    print("start find_network")
    network_triplets = triplet_df.loc[(triplet_df["Ld"] >= 0) & (triplet_df["Ld"] <= L),:]
    network_triplets.head()
    network = network_triplets.iloc[:,[0,2]]
    network = network.rename(columns={"Om": "Oc", "Ot": "Ot"})
    network = network.append(network_triplets.iloc[:,[1,2]])
    lm = ["red"]*len(network_triplets)
    lc = ["blue"]*len(network_triplets)
    lm.extend(lc)
    network["Interaction"] = lm
    network = network.drop_duplicates()
    print("end find_network")
    return network

# In[18]:


def draw_network(network):
    print("start draw_network")
    G = nx.from_pandas_edgelist(network, "Oc", "Ot", "Interaction")
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='#A0CBE2', node_size = 3000, edge_color=network["Interaction"],
        width=4, edge_cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.savefig("network.png", format="PNG")
    print("end draw_network")


# In[ ]:


def main(file, L = 3.8, oligo = False):
    print("STEP 1")
    df = format_file(file, oligo)
    print("STEP 2")
    triplet_df = triplets(df)
    print("STEP 3")
    lof = lack_of_fit(df, triplet_df)
    print("STEP 4")
    network = find_network(triplet_df, L)
    print("STEP 5")
    draw_network(network)


# In[ ]:


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

