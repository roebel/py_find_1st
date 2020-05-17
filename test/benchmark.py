#! /usr/bin/env python

import numpy as np
import timeit

setup="""
import numpy as np
import utf1st_inst_dir.utils_find_1st as utf1st
limit = 0.
rr= np.ones(100)
rr[10] = 0
"""

print("utf1st.find_1st(rr, limit, utf1st.cmp_equal)::\nruntime {:.3f}s".format(np.mean(timeit.repeat("utf1st.find_1st(rr, limit, utf1st.cmp_equal)", setup=setup))))
print("np.flatnonzero(rr==limit)[0]::\nruntime {:.3f}s".format(np.mean(timeit.repeat("np.flatnonzero(rr==limit)[0]", setup=setup))))
print("next((ii for ii, vv in enumerate(rr) if vv == limit))::\nruntime {:.3f}s".format(np.mean(timeit.repeat("next((ii for ii, vv in enumerate(rr) if vv == limit))", setup=setup))))


    

  
