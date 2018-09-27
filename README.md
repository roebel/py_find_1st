# py_find_1st #

py_find_1st  is a numpy extension that allows to find the first index into an 1D-array that
validates a boolean condition that can consist of a comparison operator and a limit value.

## Functionality ##

This extension solves the very frequent problem of finding first indices without requiring to read the full array.

The call sequence

    import numpy as np
    import utils_find_1st as utf1st
    
    limit = 0.
    rr= np.random.randn(100)
    ind = utf1st.find_1st(rr < limit, True, utf1st.cmp_equal)

and more efficiently

    ind = utf1st.find_1st(rr, limit, utf1st.cmp_smaller)

is equivalent to

    import numpy as np
    limit = 0.
    rr= np.random.randn(100)
    ind = np.flatnonzero(rr < limit)
    if len(ind) :
        ret = ind[0]
    else:
        ret = -1

## Implementation details ##

py_find_1st is written as a numpy extension making use of a templated
implementation of the find_1st function that currently supports
operating on arrays of dtypes:

    [np.float64, np.float32, np.int64, np.int32, np.bool]

Comparison operators are selected using integer opcodes with the
following meaning:

    opcode == utils_find_1st.cmp_smaller    ->  comp op: <
    opcode == utils_find_1st.cmp_smaller_eq ->  comp op: <=
    opcode == utils_find_1st.cmp_equal      ->  comp op: ==
    opcode == utils_find_1st.cmp_not_equal  ->  comp op: !=
    opcode == utils_find_1st.cmp_larger     ->  comp op: <
    opcode == utils_find_1st.cmp_larger_eq  ->  comp op: <=


## Performance ##

The runtime difference is strongly depending on the number of true cases in the array. 
If the condition is never valid runtime is the same - both implementations do not produce a valid index
and need to compare the full array - but on case that there are matches np.flatnonzero needs to
run through the full array and needs to create a result array with size that depends o the number of matches
while find_1st only produces a scalar result and only needs to compare the array until the first match is found.

Depending on the size of the   array and the number of matches the speed difference can be very significant
(easily > factor 10)


## test ##

run test/test_find_1st.py which should display "all tests passed!"

### Benchmarking ###

We can easily compare the runtime using the three lines

    In [6]: timeit ind = np.flatnonzero(rr < limit)[0]
    1.69 $\mu$s $\pm$ 24.5 ns per loop (mean $\pm$ std. dev. of 7 runs, 1000000 loops each)
    
    In [4]: timeit ind = utf1st.find_1st(rr < limit, True, utf1st.cmp_equal)
    1.13 $\mu$s $\pm$ 18.9 ns per loop (mean $\pm$  std. dev. of 7 runs, 1000000 loops each)
    
    In [5]: timeit ind = utf1st.find_1st(rr, limit, utf1st.cmp_smaller)
    270 ns $\pm$ 5.57 ns per loop (mean $\pm$ std. dev. of 7 runs, 1000000 loops each)

Which shows the rather significant improvement obtained by the last
version that does not require to perform all comparisons of the 100
elements. In the above case the second element is tested positive.
In the worst case, where no valid element is present all comparisons
have to be performed and flatnonzero does not need to create a results
array, and therefore performance should be similar. For the small array sizes we used so far
the overhead of np.flanonzero is dominating the costs as can be seen in the following.

    In [9]: limit = -1000.
    In [10]: timeit ind = np.flatnonzero(rr < limit)
    1.56 $\mu$s $\pm$ 13.8 ns per loop (mean $\pm$ std. dev. of 7 runs, 1000000 loops each)
    
    In [11]: timeit ind = utf1st.find_1st(rr<limit, True, utf1st.cmp_equal)
    1.16 $\mu$s $\pm$ 7.07 ns per loop (mean $\pm$ std. dev. of 7 runs, 1000000 loops each)
    
    In [12]: timeit ind = utf1st.find_1st(rr, limit, utf1st.cmp_smaller)
    314 ns $\pm$ 3.36 ns per loop (mean $\pm$ std. dev. of 7 runs, 1000000 loops each)

For a significantly larger array size costs become more comparable

    rr= np.random.randn(10000)
    In [13]: timeit ind = np.flatnonzero(rr < limit)
    4.87 $\mu$s $\pm$ 101 ns per loop (mean $\pm$ std. dev. of 7 runs, 100000 loops each)
    
    In [14]: timeit ind = utf1st.find_1st(rr<limit, True, utf1st.cmp_equal)
    8.95 $\mu$s $\pm$ 497 ns per loop (mean $\pm$ std. dev. of 7 runs, 100000 loops each)
    
    In [15]: timeit ind = utf1st.find_1st(rr, limit, utf1st.cmp_smaller)
    4.4 $\mu$s $\pm$ 47.9 ns per loop (mean $\pm$ std. dev. of 7 runs, 100000 loops each)

Which demonstrates that even in this case the find_1st extension is more efficient
besides if the boolean intermediate array is used in line 14.

This result is a bit astonishing as the overhead involved in passing the boolean intermediate array
into the find_1st extension seems rather large compared to the simple boolean comparison  

    In [35]: timeit ind = rr < limit
    3.31 $\mu$s $\pm$ 47.3 ns per loop (mean $\pm$ std. dev. of 7 runs, 100000 loops each)
   
The clarification of this remaining issue needs further investigation. Any comments are welcome.

## Changes ##

### Version 1.1.2 (2017-09-19) ###

  * Removed ez_setup.py that seems to be no longer maintained by setuptools maintainers.
    
### Version 1.1.1 (2017-09-19) ###

  * Use NPY_INT64/NPY_INT32 instead of NPY_INT/NPY_LONG
    such that the test does not rely on the compiler specific int sizes.

### Version 1.1.0 (2017-09-18) ###

  * fixed bug in cmp operator values that were not coherent on the python and C++ side
  * support arbitrary strides for one dimensuional arrays
  * Added test script

### Version 1.0.7 (2017-09-18) ###

  * Changed compiler test to hopefully work for MSVC under windows.

### Version 1.0.6 (2017-05-31) ###

  * Removed more non ascii elements in README.

### Version 1.0.5 (2017-05-31) ###

  * Fixed non ascii elements in README that led to problems with some
    python configurations.

### Version 1.0.4 (2017-05-31) ###

  * Fixed setup.py problems:
  on the fly generation of LONG_DESCRPTION file.

### Version 1.0.3 (2017-05-31) ###

 * Moved to github

### Version 1.0.2 (2017-05-31) ###

 * Force using c++ compiler

### Version 1.0.1 (2017-05-31) ###

 * initial release
 
## Copyright ##

Copyright (C) 2017 IRCAM

## License ##

GPL see file Copying.txt

## Author ##

Axel Roebel

