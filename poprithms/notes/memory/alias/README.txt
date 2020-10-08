The memory::alias project is for tracking and intersecting Regions of
Tensors when view-changing operators are applied to them. The class
memory::alias::Tensor has an API very similar to poplar::Tensor's, and the
2 classes behave the same way for all Tensor class member methods.

The 2 Tensor classes differ significantly in how they store aliasing
related information.

The most significant difference is the building block used to store a
set of Tensor elements. In Poplar, it is a contiguous interval. In
poprithms::memory::alias, it is a Sett, which is a generalization of an
interval which allows for striding, and nested striding. Setts have all the
usual set operations such as intersection.

In certain cases, retaining striding information can greatly speed-up
aliasing queries. An example:

a = Tensor(shape=(2,N))
b = a[0:1]
c = a[1:2]
Does b intersect c? In this case, b and c are 2 contiguous intervals, and
so 1 interval intersection is all that is required to answer the question
in poplar.


a = Tensor(shape=(10,2))
b = a[0:10, 0:1]
c = a[0:10, 1:2]
Does b intersect c? In this case, b and c each comprise N intervals of
length 1, so O(N) comparisons are required in poplar.

An example of using memory::alias is in demo_0.cpp in the test directory. 

