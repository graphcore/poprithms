

The graph being scheduled is shown below. The grey boxes denote allocations, with the number in the box denoting the allocation's size. The circles denote the nodes (ops).

![The graph](./dag.png)

The initial (random) schedule is 

```0 6 10 11 5 12 17 3 4 2 7 8 15 1 22 26 28 31 25 9 16 14 18 13 19 29 23 30 20 21 27 32 24 33 36 42 34 35 39 37 40 44 41 45 47 43 46 38 48 49```

The sequence of sum-liveness reducing rotations (shifts) looks like:

<img src="animation.gif" width="100%"/>

and the final schedule, after all rotations are applied is, 

```2 3 1 0 6 10 11 12 17 4 7 5 8 15 22 29 28 26 31 25 23 30 14 18 9 13 16 20 19 21 24 27 32 37 33 40 36 42 34 35 38 39 41 43 49 44 48 45 46 47```