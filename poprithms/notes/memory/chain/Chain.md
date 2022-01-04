## The Chain class and canonicalization

A Chain with Ops `op0` and `op1` will be expressed here as `op0 -> op1`, or as `inShape -> op0 -> op1` if the input Shape needs to be specified. 
There are currently 7 types of Op, see [chain.hpp](../../../poprithms/include/poprithms/memory/chain/type.hpp) for details.

## Canonicalization of Chains ##

The initial motivation for Chain canonicalization was for [unwinding](../unwind/Unwinding.md).
There, to verify that Tensors have the same target layout, Chains are compared for equivalence. 
Two Chains  `c0` and `c1` are (truly) equivalent if applying them to a Tensor `t` with distinct elements results in the same output Tensor. That is, for all `t`, `c0(t) = c1(t)`. 
This however is computationally expensive, requiring running full ML model size computations. 
Instead, it is preferable to compare Chains directly. 
That is, two Chains are considered equivalent if they are opwise identical: they have the same Op at every Chain position.
Canonicalization makes it more likely that 2 Chains which are (truly) equivalent are opwise identical. 

As an example, 

`DimShuffle(1 2 0) -> Reverse(0) -> DimShuffle(2 0 1)`, and `Reverse(1)`

are equivalent in the sense that the outcome of applying each of them to a Tensor is the same. 
However comparing them directly Op by Op, they are not the same. 
Canonicalization aims to reduce the longer first Chain to the second Chain, so that opwise comparison works as expected. 


The second motivation for canonicalization is that simpler Chains result in simpler Tensor expressions in Poplar, which results in reduced compile times. TODO(jn) create a task for using this work directly in PopIR.  


### Ops which do nothing can be removed

Some Ops do nothing. 
Examples are DimShuffles with the identity Permutation; 
Expands, Reshapes, Reduces, SettSamples and SettFillIntos where the output and input Shapes are the same; 
and Reverses where the Dimensions attribute is empty. 

### Contiguous Ops of the same type can sometimes be merged

Contiguous **DimShuffles** can *always* be merged, as `DimShuffle(p0) -> DimShuffle(p1)` 
is the same as  `Dimshuffle(p0 * p1)`. 
For example, `DimShuffle(1 2 0) -> DimShuffle(1 2 0) = DimShuffle(2 0 1)`.

Contiguous **Reverses** can always be merged, as 
`Reverse(dims0) -> Reverse(dims1)` and `Reverse(dims0 + dims1)` are the same. 
For example, `Reverse(1,2) -> Reverse(0,2) = Reverse(0,1)`. 

Contiguous **Reshapes**, can *always* be merged by dropping all but the final Op. 
For example,  `Reshape(2,3,5,2) -> Reshape(6,10)` is simply `Reshape(6,10)`. 
The same is true for **Expands** and **Reduces**.

Contiguous **SettSamples** can normally be merged. 
Recall that a **SettSample** is a generalized slice operation. 
If both SettSamples can be expressed as slices or subSamples, they can always be merged.
There are cases however of complex "unfactorizable" Regions which cannot be merged. 
The same is true of **SettFillInto**. 
For example, using 
`Slice(Dim, Lower:Upper:Stride)` to denote the special case where a SettSample is a 
slice or subSample in 1 dimension, 
`Slice(Dim=0, 2:6:1) -> Slice(Dim=0, 3:7:1)` is the same as `Slice(Dim=0, 3:6:1)` and 
`Slice(Dim=1, 0:12:3) -> Slice(Dim=1, 0:12:2)` is the same as `Slice(Dim=1, 0:12:6)`. 


### Op position swapping

This pass does not directly reduce the number of Ops in a Chain. 
It does however make contiguous Ops more likely to be of the same type, which makes the Op-merging pass above more likely to succeed. 

The basic idea with this pass is to sort the Ops as far as possible. 
At the moment in this project, the target ordering is lexicographic:
`DimShuffle < Expand < Reduce < Reshape < Reverse < SettFillInto < SettSample` although this choice
is arbitrary and may change. 

The ordering of Ops affects the Chain's behaviour. For example, 
`DimShuffle(0 1) -> Reverse(1)` is not the same as `Reverse(1) -> DimShuffle(0 1)`. 
To swap 2 contiguous Ops in a Chain without changing behaviour of the Chain, Ops themselves need to 
change slightly. In swapping the positions of the Ops in `Reverse(1) -> DimShuffle(0 1)` for example, the 
axis of reversal must change, to become `DimShuffle(0 1) -> Reverse(0)`. 

Further examples:

`Slice(Dim=0, 1:3:1) -> DimShuffle(1 2 0)` is the same as `DimShuffle(1 2 0) -> Slice(Dim=2, 1:3:1)`.

`(10) -> Slice(Dim=0, 1:3) -> Reverse(0)` is the same as `(10) -> Reverse(0) -> Slice(Dim=0, 7:9)`.

`DimShuffle(2 0 1) -> Reshape(5,7,6)` is the same as `Reshape(6,5,7) -> DimShuffle(1 2 0)`.


When a swap contains a Reshape, special care needs to be takenm and is only valid in certain cases. 
Examples where it is *not* possible to perform a swap are

1) `(3,2) -> Reshape(2,3) -> DimShuffle(0 1)` 
2) `(25,2,3) -> DimShuffle(0 2 1) -> Reshape(5, 5, 6)`. 

Examples of valid swaps are, 

1) `(25,3,2) -> DimShuffle(1 2 0) -> Reshape(6, 5, 5)` is the same as `(25, 3, 2) -> Reshape(5, 5, 6) -> DimShuffle(2 0 1)`
2) `(2, 2, 9, 5, 5, 49) -> DimShuffle(3 4 5 2 0 1) -> Reshape(25, 7, 7, 3, 3, 4)` is the same as `(2, 2, 9, 5, 5, 49) -> Reshape(4, 3, 3, 25, 7, 7) -> DimShuffle(3 4 5 1 2 0)`. 

More details on when a swap involving a Reshape is invalid can be found in the canonicalization implementation. 

To show how the sorting can be useful, consider the Chain 

`DimShuffle(2 0 1) -> Reverse(0) -> DimShuffle(1 2 0) -> Reverse(2)`. 

By swapping the innermost 2 Ops, one arrives at the equivalent Chain

`DimShuffle(2 0 1) -> DimShuffle(1 2 0) -> Reverse(2) -> Reverse(2)`. 

The contiguous Ops of the same type in this Chain can be merged, to get

`DimShuffle(0 1 2) -> Reverse()` 

and both of these Ops are no-Ops. So the Chain is a null-Chain. 

## Canonicalization algorithm

The passes above, along with some others, are iterated through until the Chain stops changing. 



