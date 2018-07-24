# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:25:54 2018

@author: Kristian
"""

  
def jaccard_similarity(x,y): 
    print("jplag-len:", len(x))
    print("sd-len:", len(y))
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    print("Intersection card:", intersection_cardinality)
    print("Union card:", union_cardinality)
    return intersection_cardinality/float(union_cardinality)
  