import pandas as pd
import glob
import numpy as np
import timeit
from multiprocessing import Pool
from matplotlib import pyplot as plt
import pickle

import requests, json
import pandas as pd
import glob
import timeit
import time
from multiprocessing import Pool
import numpy as np
def toc(start_time):
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
from zipfile import ZipFile
import re
import xmlschema
from pprint import pprint
import glob
# importing element tree
import lxml.etree as etree

import time
import pickle
import numpy as np
path_career='/mnt/sdb1/sandeep/Career Transitions/'
def toc(start_time):
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
start_time = timeit.default_timer()

def get_career_from_name(auth_name_to_search):
#     auth_name_to_search='monika HENZINGER'
    # url='https://api.openalex.org/authors?mailto=chowdhary_sandeep@phd.ceu.edu,page=1,filter=display_name.search:'+auth_name_to_search
    # work_groups_url='https://api.openalex.org/authors?filter=display_name.search:'+auth_name_to_search
    url='https://api.openalex.org/authors?filter=display_name.search:'+auth_name_to_search+'&sort=works_count:desc'
    headers = {
        'User-Agent': 'chowdhary_sandeep',
        'From': 'chowdhary_sandeep@phd.ceu.edu'  # This is another valid field
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # raises exception when not a 2xx response
    if response.status_code != 204:
        x= response.json()
    ## Retrieve author-id and get career
    #(I choose the first match which is usually the one with most papers) 

    #x= json.loads(response.content.decode("utf-8"))
        if 'results' in x.keys():
            if len(x['results'])>0:
                new_url__=x['results'][0]['works_api_url']

                # get first page

                cursor='*'
                works_=[]
                try:
                    response = requests.get(new_url__+'&per-page=200&cursor='+cursor,headers=headers)
                    response.raise_for_status()  # raises exception when not a 2xx response
                except:
                    response = requests.get(new_url__+'&per-page=50&cursor='+cursor,headers=headers)
                    response.raise_for_status()  # raises exception when not a 2xx response
                    
                if response.status_code != 204:
                    res= response.json()
            
                    works_.append(res)
                    it=0

                    # get all other pages
                    while not(res['meta']['next_cursor'] is None):
                        it+=1
                        cursor=res['meta']['next_cursor']
                        try:
                            response = requests.get(new_url__+'&per-page=200&cursor='+cursor,headers=headers)
                            response.raise_for_status()  # raises exception when not a 2xx response

                        except:
                            response = requests.get(new_url__+'&per-page=50&cursor='+cursor,headers=headers)
                            response.raise_for_status()  # raises exception when not a 2xx response
                        if response.status_code != 204:
#                             res= response.json()                            
                            res = json.loads(response.content.decode("utf-8"))
    
                            if len(res['results'])>0:
                                works_.append(res)

                    career_=[]
                    for it in range(len(works_)):
                        career_=career_+(works_[it]['results'])
                    return career_
                else:
                    return 'NA'
            else:
                return 'NA'
        else:
            return 'NA'
    else:
        return 'NA'
    