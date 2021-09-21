# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from shift_util import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def createAnimation(livenesses, shifts, gifName="shiftingDemo.gif"):
    """
    Creates an animimation such as the ones linked to in the Shift.md introduction
    """

    # maximum over all of the schedules and over all schedule indices
    maxLiveness = max([max(p) for p in livenesses])

    fig, ax = plt.subplots()

    ystart = 0
    yend = maxLiveness * 1.4
    ax.set_ylim([ystart, yend])

    ax.set_xlabel("schedule index")
    ax.set_ylabel("liveness")

    x, y = getLivenessPoints(livenesses[0])
    ax.plot(x, y, color='k', label='initial liveness profile')

    bar0 = ax.fill_between(x=x,
                           y1=y,
                           y2=y,
                           color="#E84A5F",
                           alpha=0.8,
                           label='sub-schedule 0')

    bar1 = ax.fill_between(x=x,
                           y1=y,
                           y2=y,
                           color="#355CFD",
                           alpha=0.8,
                           label='sub-schedule 1')

    line0 = ax.fill_between(x=x,
                            y1=y,
                            y2=y,
                            color="#99B898",
                            alpha=0.75,
                            label='current liveness profile')

    liveness_text = ax.text(
        x=0.05,
        y=0.9,
        s="current sum-liveness = %d" % (sum(livenesses[0]), ),
        transform=ax.transAxes,
        horizontalalignment='left')

    ax.legend()

    def animate(frameNumber):
        """
        Update the animation for frame #frameNumber.
        """

        shiftIndex = (frameNumber // 2)
        shift = shifts[shiftIndex]

        b00 = shift[0]
        b01 = b00 + shift[2]

        delta = shift[1] - shift[0]
        b10 = b00 + (delta > 0) * shift[2]
        b11 = b10 + delta

        if (frameNumber % 2 == 1):
            b00 += delta
            b01 += delta
            b10 += (1 - 2 * (delta > 0)) * shift[2]
            b11 += (1 - 2 * (delta > 0)) * shift[2]

        bar0.set_verts([[[b00, 0], [b00, maxLiveness], [b01, maxLiveness],
                         [b01, 0]]])

        bar1.set_verts([[[b10, 0], [b10, maxLiveness], [b11, maxLiveness],
                         [b11, 0]]])

        livenessIndex = (frameNumber + 1) // 2
        liveness = livenesses[livenessIndex]
        line0.set_verts([getLivenessesPoly(liveness)])
        liveness_text.set_text("current sum-liveness = %d" % (sum(liveness)))

        return bar0, bar1, line0

    frames = range(2 * len(shifts))

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  blit=True,
                                  interval=1000,
                                  frames=frames)

    
    #frames per second 
    fps = 3
    writer = animation.FFMpegWriter(fps=fps,
                                    metadata=dict(artist='Me'),
                                    bitrate=2800)

    ani.save(gifName, writer=writer)


def graphvizDag(jsonGraph, filename="dagger"):

    import graphviz

    tiger = '#FC6A03'
    honey = '#FFA500'
    gray = 'gray'

    #Insert all the (orange) DAG edges.
    g = graphviz.Digraph('G', filename=filename, format="png")
    g.attr('node', color=honey)
    g.attr('node', style='filled')
    g.attr('node', shape='doublecircle')
    g.attr('edge', color=tiger)
    for op in jsonGraph['ops']:
        x = '%d' % (op['address'], )
        g.node(x)  #, '%d'%(allocs[a],))
        for out in op['outs']:
            g.edge(x, '%d' % (out, ))

    #Insert of all [gray] allocations and their edges.
    allocs = getAllocs(jsonGraph)
    g.attr('node', color=gray)
    g.attr('node', shape='box')
    g.attr('edge', color=gray)
    g.attr('edge', arrowhead='none')
    g.graph_attr['rankdir'] = 'TB'
    for op in jsonGraph['ops']:
        for a in op['allocs']:
            x = '%d_' % (a, )
            g.node(x, '%d' % (allocs[a], ))
            g.edge(x, '%d' % (op['address'], ))

    g.view()


def getScheduleString(schedule):
    return ' '.join(map(str, schedule))


def makeMarkdown(graph, livenesses, shifts, initialSchedule, finalSchedule,
                 bdRead, bdWrite):

    import os

    print("Creating graphviz dag")
    graphvizDag(graph, os.path.join(bdWrite, "dag"))
    # we only wany the file dag.png, not the text file 'dag' which is also created 
    os.remove(os.path.join(bdWrite, "dag"))

    print("Creating animation")
    createAnimation(livenesses, shifts, os.path.join(bdWrite, "animation.gif"))

    s = r"""

The graph being scheduled is shown below. The grey boxes denote allocations, with the number in the box denoting the allocation's size. The circles denote the nodes (ops).

![The graph](./dag.png)

The initial (random) schedule is 

```""" + getScheduleString(initialSchedule) + r"""```

The sequence of sum-liveness reducing rotations (shifts) looks like:

<img src="animation.gif" width="100%"/>

and the final schedule, after all rotations are applied is, 

```""" + getScheduleString(finalSchedule) + "```"

    x = open(os.path.join(bdWrite, "demo.md"), 'w')
    x.write(s)

    print(s)


# we prefer
# <img src="animation.gif" width="100%"/>
# over
# ![This is a test 1 ](./animation.gif)
# as it allows us to scale to the full column width.
