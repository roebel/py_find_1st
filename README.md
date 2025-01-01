# py_find_1st

py_find_1st is a numpy extension that allows to find the first index into an 1D-array that
validates a boolean condition that can consist of a comparison operator and a limit value.

## Functionality

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

## Implementation details

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


## Performance

The runtime difference is strongly depending on the number of true cases in the array. 
If the condition is never valid runtime is the same - both implementations do not produce a valid index
and need to compare the full array - but on case that there are matches np.flatnonzero needs to
run through the full array and needs to create a result array with size that depends o the number of matches
while find_1st only produces a scalar result and only needs to compare the array until the first match is found.

Depending on the size of the   array and the number of matches the speed difference can be very significant
(easily > factor 10)


## test

run test/test_find_1st.py which should display "all tests passed!"

### Benchmarking

We can easily compare the runtime for the setup displayed before executing the three lines

<pre><code>
In [6]: timeit ind = np.flatnonzero(rr < limit)[0]
1.69&mu;s &plusmn; 24.5ns per loop (mean &plusmn; std. dev. of 7 runs, 1000000 loops each)
    
In [4]: timeit ind = utf1st.find_1st(rr limit, True, utf1st.cmp_equal)
1.13&mu;s &plusmn; 18.9ns per loop (mean &plusmn;  std. dev. of 7 runs, 1000000 loops each)
    
In [5]: timeit ind = utf1st.find_1st(rr, limit, utf1st.cmp_smaller)
270ns &plusmn; 5.57ns per loop (mean &plusmn; std. dev. of 7 runs, 1000000 loops each)
</code></pre>


It shows the rather significant improvement obtained by the last
version that does not require to perform all comparisons of the 100
elements. In the above case the second element is tested positive.
In the worst case, where no valid element is present all comparisons
have to be performed and flatnonzero does not need to create a results
array, and therefore performance should be similar. We can benchmark this case by means of changing
the limit such that it does never fit. For the small array sizes we used so far
the overhead of np.flanonzero is dominating the costs as can be seen in the following.

<pre><code>
In [9]: limit = -1000.
In [10]: timeit ind = np.flatnonzero(rr < limit)
1.56&mu;s &plusmn; 13.8ns per loop (mean &plusmn; std. dev. of 7 runs, 1000000 loops each)

In [11]: timeit ind = utf1st.find_1st(rr < limit, True, utf1st.cmp_equal)
1.16&mu;s &plusmn; 7.07ns per loop (mean &plusmn; std. dev. of 7 runs, 1000000 loops each)

In [12]: timeit ind = utf1st.find_1st(rr, limit, utf1st.cmp_smaller)
314ns &plusmn; 3.36ns per loop (mean &plusmn; std. dev. of 7 runs, 1000000 loops each)
</code></pre>

For a significantly larger array size costs become more comparable

<pre><code>
rr= np.random.randn(10000)
In [13]: timeit ind = np.flatnonzero(rr < limit)
4.87&mu;s &plusmn; 101ns per loop (mean &plusmn; std. dev. of 7 runs, 100000 loops each)

In [14]: timeit ind = utf1st.find_1st(rr < limit, True, utf1st.cmp_equal)
8.95&mu;s &plusmn; 497ns per loop (mean &plusmn; std. dev. of 7 runs 100000 loops each)

In [15]: timeit ind = utf1st.find_1st(rr, limit, utf1st.cmp_smaller)
4.4&mu;s &plusmn; 47.9ns per loop (mean &plusmn; std. dev. of 7 runs, 100000 loops each)
</code></pre>

Which demonstrates that even in this case the find_1st extension is more efficient besides if the boolean intermediate array is used in line 14. In that case the comparison of the full array is actually performed twice.

This result is nevertheless a bit higher than expected from the sum of the simple boolean comparison

<pre><code>
In [35]: timeit ind = rr < limit
3.31&mu;s &plusmn; 47.3ns per loop (mean &plusmn; std. dev. of 7 runs, 100000 loops each)
</code></pre>

and the find_1st operation in line 15 above, which would result in 7.7&mu;s. The clarification of this remaining issue needs further investigation. Any comments are welcome.

## Changes

### Version 1.1.7rc5 (2025-01-01)

  * Updated build system to support numpy 2.0 and to avoid running setup.py directly.
  * Fixed math display in README.

### Version 1.1.6 (2023-10-28)

  * fixed test script for numpy > 1.20
  * fixed Makefile under macosx ARM64

### Version 1.1.5 (2021-02-02)

  * fixed problems with numpy dependency handling (thanks to xmatthias).
    Now use oldest-supported-numpy instead of using the most recent numpy.

### Version 1.1.4 (2019-08-04)

  * added support for automatic installation of requirements
  * add and support pre-release tags in the version number
  * use hashlib to calculate the README checksum.
  * support testing via `make check`

### Version 1.1.3 (2018-10-05)

  * Removed setting stdlib for clang in setup.py - the default should do just fine.
 
### Version 1.1.2 (2018-09-28)

  * Removed ez_setup.py that seems to be no longer maintained by setuptools maintainers.
    
### Version 1.1.1 (2017-09-19)

  * Use NPY_INT64/NPY_INT32 instead of NPY_INT/NPY_LONG
    such that the test does not rely on the compiler specific int sizes.

### Version 1.1.0 (2017-09-18)

  * fixed bug in cmp operator values that were not coherent on the python and C++ side
  * support arbitrary strides for one dimensional arrays
  * Added test script

### Version 1.0.7 (2017-09-18)

  * Changed compiler test to hopefully work for MSVC under windows.

### Version 1.0.6 (2017-05-31)

  * Removed more non ascii elements in README.

### Version 1.0.5 (2017-05-31)

  * Fixed non ascii elements in README that led to problems with some
    python configurations.

### Version 1.0.4 (2017-05-31)

  * Fixed setup.py problems:
  on the fly generation of LONG_DESCRIPTION file.

### Version 1.0.3 (2017-05-31)

 * Moved to github

### Version 1.0.2 (2017-05-31)

 * Force using c++ compiler

### Version 1.0.1 (2017-05-31)

 * initial release
 
## Copyright

Copyright (C) 2017 IRCAM

## License 

GPL see file Copying.txt

## Author

Axel Roebel

