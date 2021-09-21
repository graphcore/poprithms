

The graph being scheduled is shown below. The grey boxes denote allocations, with the number in the box denoting the allocation's size. The circles denote the nodes (ops).

![The graph](./dag.png)

The initial (random) schedule is 

```0 10 9 5 4 1 2 6 15 7 12 16 17 3 21 8 18 26 11 13 23 20 28 24 25 19 29 14 27 22```

The sequence of sum-liveness reducing rotations (shifts) looks like:

<img src="animation.gif" width="100%"/>

and the final schedule, after all rotations are applied is, 

```0 3 4 10 1 7 5 12 18 9 26 16 17 6 21 8 11 2 14 13 15 19 22 23 25 20 27 28 29 24```