The algorithm works as follows: 

Start by initializing the schedule as any feasible schedule. Then, looks for valid shifts of contiguous chunks in the schedule. Example

   a  b  c  d  e  f   (i)
      =======  ----

 to

   a  e  f  b  c  d   (ii)
      ----  =======  

is a valid shift if and only if all topological constraints are satisfied after in the resulting schedule (ii). 

If the total sum over all allocations of (liveness*size) decreases, the shift is applied to the schedule. Otherwise it is not, and a new shift is proposed. For example, if the Ops a, b, c, d, e, and f have allocations

a: A
b: A, B
c: A, B, C
d: A, B, C, D
e: A, E
f: E

and all alocations have size 1, then the liveness for (i) each allocation is 

A : 5 
B : 3
C : 2
D : 1
E : 2

and for (ii) they are

A : 6
B : 3
C : 3
D : 1
E : 2

so the total liveness increases from 13 to 15, and so this shift is rejected. 

Technically, this is annealing with zero temperature (T=0). True annealing would allow some liveness-increasing shifts (T>0) to be accepted, and then taper down to T=0 as the algorithm proceeds. But running with T=0 from the start does a good job in current experiments, so for now there is no option for T>0. 

How are shifts proposed? Shifts involving a single Op are first considered, such as 

   a  b  c  d  e  f 
      =======  -

 to

   a  e  b  c  d  f 
      -  =======  

We call such shifts involving a single Op, 1-shifts. when there are no more 1-shifts which decrease total liveness, 2-shifts are considered. 

The algorithm climbs up and down the "n-shift ladder", until there are no shifts of any size which decrease liveness. At this point the algorithm terminates. 

For many examples, the local minimum obtained is the global minimum, see for example recompute.cpp and the diagram at link (TODO(jn)). There are however random graphs for the which the total liveness of the final local minimum depends on how the algorithm is seeded (oh yes, I forgot to mention, there is randomness involved in the order in which shifts are considered). This means that the local minima are not global minima - the algorithm is not perfect!

More discussion coming soon. Examples can be found test directory. 
