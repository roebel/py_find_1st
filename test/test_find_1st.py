#! /usr/bin/env python

import numpy as np

import utf1st_inst_dir.utils_find_1st as utf1st

for type in [np.float64, np.float32, np.int64, np.int32]:

    arr = np.arange(21, dtype=type)
    
    ind = utf1st.find_1st( arr, 10, utf1st.cmp_larger_eq) 
    if arr[ind] != 10 :
        raise RuntimeError("find_1st failed type {0}, stride 1".format(str(type)) )

    ind = utf1st.find_1st( arr, 10, utf1st.cmp_equal) 
    if arr[ind] != 10 :
        raise RuntimeError("find_1st failed type {0}, stride 1".format(str(type)) )

    ind = utf1st.find_1st( arr[::2], 10, utf1st.cmp_larger_eq) 
    if arr[::2][ind] != 10 :
        raise RuntimeError("find_1st failed type {0}, stride 2".format(str(type)) )

    ind = utf1st.find_1st( arr[-1::-1], 10, utf1st.cmp_smaller_eq) 
    if arr[-1::-1][ind] != 10 :
        raise RuntimeError("find_1st failed type {0}, stride -1".format(str(type)) )
    ind = utf1st.find_1st( arr[-1::-2], 10, utf1st.cmp_smaller_eq) 
    if arr[-1::-2][ind] != 10 :
        raise RuntimeError("find_1st failed type {0}, stride -2".format(str(type)) )
    

arr = np.concatenate((np.zeros(10, dtype=bool), np.ones(10,dtype=bool)))
    
ind = utf1st.find_1st( arr, 1, utf1st.cmp_equal) 
if (not arr[ind])  or arr[ind-1]  :
    raise RuntimeError("find_1st failed type bool, stride 1" )
ind = utf1st.find_1st( arr[::2], 1, utf1st.cmp_equal) 
if (not arr[::2][ind])  or arr[::2][ind-1]  :
    raise RuntimeError("find_1st failed type bool, stride 2" )
ind = utf1st.find_1st( arr[-1::-1], 1, utf1st.cmp_equal) 
if (not arr[-1::-1][ind])  or arr[-1::-1][ind-1]  :
    raise RuntimeError("find_1st failed type bool, stride -1" )
ind = utf1st.find_1st( arr[-1::-2], 1, utf1st.cmp_equal) 
if (not arr[-1::-2][ind])  or arr[-1::-2][ind-1]  :
    raise RuntimeError("find_1st failed type bool, stride -2" )


print("all tests passed!")
    

  
