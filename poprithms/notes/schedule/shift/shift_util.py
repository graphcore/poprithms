# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Note: do not import python libraries in this file, to keep testing stable.


def getLivenesses(filename):
    """
    Expects a file with lines of the form 
    `((0,0,0,v,0,0,0),(0,0,0,w,0,0,0))`
    """
    livenesses = []
    with open(filename) as filly:
        lines = filly.readlines()
        for l in lines:
            frags = l.strip().replace("'", "").replace(" ", "").replace(
                "))", "").replace("((", "").split("),(")
            livenesses.append([float(x.split(',')[3]) for x in frags[0:-1]])
    return livenesses


def getSchedule(filename):
    """
    Expects a single line of the form 
    (1,4,3,2,6,4,5,0,8)
    """
    with open(filename) as filly:
        line = filly.readlines()[0]
        line = line.replace('(', ' ').replace(')', ' ').replace(',', ' ')
        return [int(x) for x in line.strip().split()]


def getGraph(filename):
    try:
        import json
    except:
        print("Failed to import json, will return `None`")
        return None

    graph = open(filename)
    jGraph = json.load(graph)
    return jGraph


def getAllocs(graph):
    m = {}
    for x in graph['allocs']:
        m[x['address']] = x['weight'][3]
        for i in [0, 1, 2, 4, 5, 6]:
            if x['weight'][i] is not 0:
                raise RuntimeException(
                    "Expected all values except the one at index 3 to be 0")

    return m


def getShifts(filename):
    """
    Each line is the file should be of the form:
    'start0:7 start1:9 nToShift:1'
    """
    lines = open(filename).readlines()
    shifts = []
    for l in lines:
        l = l.strip().split()
        if l:
            shifts.append([int(x.split(':')[-1]) for x in l])
    return shifts


def getLivenessPoints(liveness):
    """
    histogram points for the liveness plot. It will be used for a plot like:

     ^
     |       *
     |   * * *
     |  ** ***
     | *********
     +-------------->
      schedule index.

    
    For example, if the livenesses are [3,5], the points will be,
    [[0,3],[1,3],[1,5],[2,5]]
    
    The points are connected alternatively with horizontal and vertical lines. 
    """
    xs = []
    ys = []
    for op in range(len(liveness)):
        if op == 0:
            xs.append(0)
        else:
            xs.append(xs[-1])

        xs.append(xs[-1] + 1)
        ys.append(liveness[op])
        ys.append(liveness[op])

    assert len(xs) == len(ys)
    assert len(xs) == 2 * len(liveness)

    return xs, ys


def getLivenessesPoly(liveness):
    """
    Points in the liveness histogram. 
    For example, if the livenesses are [3,5], the polygon will be,
    [[0,0],[0,3],[1,3],[1,5],[2,5],[2,0]]
    """

    poly = []
    x, y = getLivenessPoints(liveness)
    for i in range(len(x)):
        poly.append([x[i], y[i]])
    poly.insert(0, [poly[0][0], 0])
    poly.append([poly[-1][0], 0])
    return poly
