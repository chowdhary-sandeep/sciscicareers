import pandas as pd
from collections import Counter,defaultdict,OrderedDict
from itertools import combinations
import numpy as np 
from multiprocessing import Pool,cpu_count
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.special import binom
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np
import time


#sns.set_style('white')
sns.set_context('talk')#, font_scale=1.5)
import matplotlib as mpl
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


def get_svs(df,dict_auth_N_ordx,dict_authwc_after_year_x_PHY,dict_firstpub,min_order=2,max_order=10,approximate=True):

    observables = df.groupby('b')['a'].apply(lambda x: tuple(sorted(x))).tolist()
    print(len(observables))
    
    if max_order!=0:
        max_order = min(max_order,max(map(len,observables)))
    else:
        max_order = max(map(len,observables))
#     max_order=10
    print(max_order)

    s_groups = []

    neigh_set_a_sub = dict(df.groupby('a')['b'].apply(list).reset_index().values) 
    N = df.b.nunique()
    na = df.a.nunique()
    
    if approximate: deg_a = Counter(df.a)

    significant_cores3 = []
    nonsignificant_cores3 = []
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
        print('drop')
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
                
                # MAKE DICTIONARY
                # YEAR_YOUNGEST=np.max ([dict_first_paper_year[x] for x in g])
    
                
            #p.close()

            #p.close()
            #groups = {i: groups[i] for i in groups if i not in drop}
            #for i in drop: del groups[i]
        print(len(groups))
        print('dropped')

        #print(len(groups))
        p = Pool(processes=cpu_count())
#         p = Pool(processes=8)
        if not approximate:

            pvalues = dict(zip(groups,p.map(pvalue_intersect,zip(groups,[neigh_set_a_sub]*len(groups),[N]*len(groups)))))
        else:
#             pvalues = dict(zip(groups,p.map(p_appr,[tuple([groups[i],sum([dict_auth_N_ordx[x] for x in i])])+tuple([deg_a[ii] for ii in i]) for i in groups])))


            # max_first_year: groups

#     dict_authworkcount_after_year={'openalex/a21312312':{'1997':28,......}}
#  modify it as follows---- dict_authworkcount_after_year
    
            pvalues = dict(zip(groups,p.map(p_appr,[tuple([groups[i],sum([dict_auth_N_ordx[x] for x in i])])+
                                                    tuple([dict_authworkcount_after_year[ii][dict_YEAR_YOUNGEST_START[str(i)]] for ii in i]) for i in groups])))

            #tuple_to_validate_order = partial(tuple_to_validate,groups=groups,N=N,deg_a=deg_a)
            #params = p.map(tuple_to_validate_order,groups)
            #pvalues = dict(zip(groups,p.map(p_appr,params)))
            
        p.close()

        n_possible = binom(na,order)
        bonf = 0.01/n_possible

        temp_df = pd.DataFrame(pvalues.items())
        #print(temp_df)
        try:
            temp_df.columns = ['group','pvalue']
        except: temp_df = pd.DataFrame(columns=['group','pvalue'])
        #print(temp_df.pvalue.min())
        ps = np.sort(temp_df.pvalue)
        k = np.arange(1,len(ps)+1)*bonf
        try: fdr = k[ps<k][-1] 
        except: fdr = 0
        if approximate: temp_df['w'] = temp_df.group.apply(lambda x: groups[x])
        temp_df['fdr'] = temp_df['pvalue']<bonf
        temp_df['ni'] = temp_df['group'].apply(lambda x: tuple([deg_a[ii] for ii in x]))
        temp_df['N'] = temp_df['group'].apply(lambda i: sum([dict_auth_N_ordx[x] for x in i]))
#         svh_dfs.append(temp_df.query('fdr'))
        
        significant_cores3=temp_df.query('fdr')
        value=False;
        nonsignificant_cores3=temp_df.query('fdr== @value')
        PAPERS_TOGETHER=2
        significant_cores3_sample=significant_cores3[significant_cores3['w']>PAPERS_TOGETHER]
        nonsignificant_cores3_sample=nonsignificant_cores3.sample(n=5*significant_cores3_sample.shape[0], random_state=12)
    
        path_career='/mnt/sdb1/sandeep/openalex_ACTIV/'
        import pickle
        with open(path_career+'significant_cores3_size'+str(order)+'_v1.pkl', 'wb') as f:
            pickle.dump(significant_cores3_sample, f)    
        with open(path_career+'significant_non_cores3_size'+str(order)+'_v1.pkl', 'wb') as f:
            pickle.dump(nonsignificant_cores3_sample, f)    
        print(len(groups_higher_order),temp_df.fdr.sum())

        s_groups_order = temp_df.query('fdr').group.tolist()

        s_groups.extend(s_groups_order)
#     significant_cores3=pd.concat(significant_cores3)
#     nonsignificant_cores3=pd.concat(nonsignificant_cores3)
#     PAPERS_TOGETHER=2
#     significant_cores3_sample=significant_cores3[significant_cores3['w']>PAPERS_TOGETHER]
#     nonsignificant_cores3_sample=nonsignificant_cores3.sample(n=3*significant_cores3_sample.shape[0], random_state=12)

#     return significant_cores3_sample,nonsignificant_cores3_sample