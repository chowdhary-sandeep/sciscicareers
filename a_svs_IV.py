import pandas as pd
from collections import Counter,defaultdict,OrderedDict
from itertools import combinations
import numpy as np 
from multiprocessing import Pool,cpu_count
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.special import binom
from functools import partial
# %matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np
import time

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
#sns.set_style('white')
sns.set_context('talk')#, font_scale=1.5)
# import matplotlib as mpl
# mpl.rcParams.update({'text.usetex': False})
from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()

def p_over(t):
    w,n,na,nb = t
    return st.hypergeom.sf(w-1,n,na,nb)

def p_appr(t):
    n12 = t[0]
    n = t[1]
    ns = np.array(t[2:])
    order = len(ns)
    p = st.binom.sf(n12-1,p=np.prod(ns/n),n=n)
    if p>=0:
        return p
    else:
        print(n12,n,ns)
        return p
    
def tpr(true,pred):
    return len(set(pred).intersection(true))/len(true)

def fdr(true,pred):
    try:
        return len(set(pred).difference(true))/len(pred)
    except: return np.nan
def jaccard(true,pred):
    return len(set(true).intersection(pred))/len(set(true).union(pred))

def expand(x,order):
    return tuple(combinations(x,order))

def expand_filter(x,order,drop):
    return list(set(combinations(x,order)).difference(drop))

def tuple_to_validate(x,groups,N,deg_a):
    return tuple([groups[x],N])+tuple([deg_a[ii] for ii in x])


def get_N(group,dict_neigh2,dict_auth_to_papers):
    authors_depth2=set([])
    
    for auth in group:
#         authors_depth2=authors_depth2+dict_neigh2[auth]
        authors_depth2=authors_depth2.union(dict_neigh2[auth])
#     authors_depth2=set(authors_depth2)
    
    papers=set([])
    for auth in authors_depth2:
        papers=papers.union(dict_auth_to_papers[auth])
#     unique_papers=set(papers)
    return len(papers)

def get_N2(group,dict_auth_papers_order_5):
#     print(group_short[0][0])
    x=[]
    for auth in group:
        x=x+dict_auth_papers_order_5[auth]
    return len(set(x))
    
def get_svs(df,dict_auth_papers_order_5,min_order=2,max_order=0,approximate=True):

    observables = df.groupby('b')['a'].apply(lambda x: tuple(sorted(x))).tolist()
    print(len(observables))
    
    if max_order!=0:
        max_order = min(max_order,max(map(len,observables)))
    else:
        max_order = max(map(len,observables))
        
    print(max_order)

    s_groups = []

    neigh_set_a_sub = dict(df.groupby('a')['b'].apply(list).reset_index().values)
    N = df.b.nunique()
    na = df.a.nunique()
    
    if approximate: deg_a = Counter(df.a)

    svh_dfs = []
    
    groups_higher_order = {}
    t_ic = time.time();
    for order in list(range(min_order,max_order+1))[::-1]:
        t_oc = time.time();
        print(order, '---- ',str(round(t_oc-t_ic,2)))

        expand_order = partial(expand,order=order) 
        
        order_obs = filter(lambda x: len(x)>=order,s_groups)
        #p = Pool(processes=cpu_count())
        B = (i for ii in map(expand_order,order_obs) for i in ii)
        drop = Counter(B)
        #p.close()
        
        expand_filter_order = partial(expand_filter,order=order,drop=drop)
#         print('drop')
        if not approximate:
        
            groups = set()
            for l in (map(lambda x: tuple(combinations(x,order)), filter(lambda x: len(x)>=order,observables))): 
                for g in l: groups.add(g)
            groups = groups.difference(drop)
                    
        else:
            
            
            order_obs = filter(lambda x: len(x)>=order,observables)
            #p = Pool(processes=cpu_count())
            B = (i for ii in map(expand_order,order_obs) for i in ii)
            groups = Counter(B)
            for g in drop: del groups[g]
            #p.close()

            #p.close()
            #groups = {i: groups[i] for i in groups if i not in drop}
            #for i in drop: del groups[i]
        print('number of groups=',len(groups))

#         print('dropped')

#         get_N2_groups = partial(get_N2,dict_auth_papers_order_5=dict_auth_papers_order_5)
#         p = Pool(processes=cpu_count())
#         p = Pool(processes=4)
#         n2s = dict(zip(groups,p.map(get_N2_groups,groups)))
#         p.close()
        

        # no parallelization version- Sandeep
#         n2s= dict(zip(groups,get_N2_groups(groups)))
        #print(len(groups))
        n2s={i:get_N2(i,dict_auth_papers_order_5) for i in groups}
        params = [tuple([groups[i], n2s[i]] )+tuple([deg_a[ii] for ii in i])  for i in groups]
        
#         p = Pool(processes=cpu_count())
#         p = Pool(processes=2)
        if not approximate:

            pvalues = dict(zip(groups,p.map(pvalue_intersect,zip(groups,[neigh_set_a_sub]*len(groups),[N]*len(groups)))))
        else:
#method 1            
#             pvalues = dict(zip(groups,p.map(p_appr,[tuple([groups[i],N])+tuple([deg_a[ii] for ii in i]) for i in groups])))
            #N_depth2=get_N(groups,dict_neigh2,dict_auth_to_papers)
#             print('group size=',N_depth2,end='\r')
#method 2
#             pvalues = dict(zip(groups,p.map(p_appr,[tuple([groups[i],get_N(i,dict_auth_papers_order_5)])+tuple([deg_a[ii] for ii in i]) for i in groups])))
    
#method 3    
#             pvalues = dict(zip(groups,p.map(p_appr, params)))
            pvalues={i[]:p_appr(i) for i in params}
    
    
            #tuple_to_validate_order = partial(tuple_to_validate,groups=groups,N=N,deg_a=deg_a)
            #params = p.map(tuple_to_validate_order,groups)
            #pvalues = dict(zip(groups,p.map(p_appr,params)))
            
#         p.close()

        n_possible = binom(na,order)
        bonf = 0.01/n_possible

        temp_df = pd.DataFrame(pvalues.items())
        #print(temp_df)
        try:
            temp_df.columns = ['group','pvalue']
        except: temp_df = pd.DataFrame(columns=['group','pvalue'])
        #print(temp_df.pvalue.min())
        ps = np.sort(temp_df.pvalue)
        #k = np.arange(1,len(ps)+1)*bonf
        #try: fdr = k[ps<k][-1] 
        #except: fdr = 0
        if approximate: temp_df['w'] = temp_df.group.apply(lambda x: groups[x])
        temp_df['fdr'] = temp_df['pvalue']<bonf
        temp_df['ni'] = temp_df['group'].apply(lambda x: tuple([deg_a[ii] for ii in x]))

        svh_dfs.append(temp_df.query('fdr'))
        
        print(len(groups_higher_order),temp_df.fdr.sum())

        #s_groups_order = temp_df.query('fdr').group.tolist()

        #s_groups.extend(s_groups_order)

    return pd.concat(svh_dfs)