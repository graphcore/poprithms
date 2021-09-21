

The graph being scheduled is shown below. The grey boxes denote allocations, with the number in the box denoting the allocation's size. The circles denote the nodes (ops).

![The graph](./dag.png)

The initial (random) schedule is 

```0 1 8 14 2 3 9 26 4 27 20 32 21 10 15 16 5 17 33 11 6 44 22 28 38 29 45 46 23 7 24 50 25 18 39 19 34 40 35 36 47 30 12 37 31 51 13 41 42 52 43 53 48 49 54 55 56 57 58 59 60 61 62 63```

The sequence of sum-liveness reducing rotations (shifts) looks like:

<img src="animation.gif" width="100%"/>

and the final schedule, after all rotations are applied is, 

```0 8 9 10 11 12 1 14 2 3 20 21 22 23 24 26 4 5 32 33 34 35 36 37 38 39 40 41 42 43 6 7 44 45 46 47 50 51 48 52 53 54 55 49 56 57 58 59 27 28 29 30 31 60 25 61 15 16 17 18 19 62 13 63```