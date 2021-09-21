

The graph being scheduled is shown below. The grey boxes denote allocations, with the number in the box denoting the allocation's size. The circles denote the nodes (ops).

![The graph](./dag.png)

The initial (random) schedule is 

```0 1 2 4 9 19 5 10 6 13 12 3 39 8 17 35 40 7 27 28 18 37 56 26 55 22 58 53 75 11 67 24 36 20 57 65 76 15 16 50 31 54 38 14 74 30 21 34 25 46 62 29 51 33 41 49 45 23 72 47 32 61 48 78 42 44 71 83 85 68 81 70 66 60 43 63 69 59 64 82 52 79 88 73 80 87 91 84 77 89 86 90 92 93 94```

The sequence of sum-liveness reducing rotations (shifts) looks like:

<img src="animation.gif" width="100%"/>

and the final schedule, after all rotations are applied is, 

```0 2 1 3 4 10 9 22 45 46 70 21 43 44 69 82 19 39 40 67 20 42 41 68 81 88 7 8 17 36 35 65 18 37 38 66 80 15 32 31 63 16 33 34 64 79 87 91 5 6 14 13 27 28 55 56 75 57 58 76 85 29 60 59 77 30 62 61 78 86 90 12 26 53 54 74 25 51 52 73 84 11 24 23 47 48 71 49 50 72 83 89 92 93 94```