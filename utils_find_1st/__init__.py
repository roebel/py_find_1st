from __future__ import absolute_import
from .find_1st import find_1st 

version_str = "1.1.7"
version = tuple(( int(ss) if ss.isdigit() else ss for ss in version_str.split('.')))

cmp_smaller    = -2
cmp_smaller_eq = -1
cmp_equal      = 0
cmp_larger_eq  = 1
cmp_larger     = 2
cmp_not_equal  = 3


    
