#include <map>
#include <set>
#include <sstream>
#include <unordered_set>

#include <poprithms/schedule/supercon/error.hpp>
#include <poprithms/schedule/supercon/graph.hpp>
#include <poprithms/schedule/supercon/logging.hpp>
#include <poprithms/util/unisort.hpp>

namespace poprithms {
namespace schedule {
namespace supercon {

class Graph {

  // number of Ops in the Graph
  uint64_t nOps;

  // The forward edges in the Graph
  Edges edgesForward;
  // The backward edges in the Graph
  Edges edgesBackward;

  // The Ops which have all their inputs scheduled, but have not themselves
  // yet been scheduled.
  std::vector<OpId> ready;

  // All Ops which have at some point been in the "ready" waiting vector.
  std::vector<bool> haveBeenReady;

  // The number of inputs each Op has had scheduled. Only when the number of
  // inputs which have been scheduled is equal to the total number of
  // inputs, can an Op be placed in "ready" from where it is put into the
  // schedule.
  std::vector<uint64_t> nInsScheduled;

  // pairs[i] is all the Pairs that Op i is the first member of i. That is,
  // for all x \in pairs[i] x[0] = i.
  std::vector<std::vector<Pair>> pairs;

  // If key[0] precedes key[1] in the schedule, then v[0] must precede v[1]
  // for all v in implications[key].
  std::map<Pair, std::vector<Pair>> implications;

  // Pairs correspond to constraints in the Graph. Pairs which are ready to
  // be converted into Graph constraints, but have yet been converted, are
  // temporarilty stored in readyPairs (analogous to "ready" for Ops)
  std::vector<Pair> readyPairs;

  // A record of Pairs which are or have been in readyPairs is kept here:
  std::set<Pair> haveBeenReadyPairs;

  void insertEdge(OpId from, OpId to) {
    edgesForward[from].push_back(to);
    edgesBackward[to].push_back(from);
  }

  void registerPair(Pair p) {
    if (haveBeenReadyPairs.count({p[0], p[1]}) == 0) {
      readyPairs.push_back(p);
      haveBeenReadyPairs.insert(p);
    }
  }

  void insertReady(OpId id) {
    ready.push_back(id);
    haveBeenReady[id] = true;
    for (auto p : pairs[id]) {
      if (p[0] != id) {
        throw error("Logic error in outline::linear::Graph::setSchedule");
      }
      if (!haveBeenReady[p[1]]) {
        registerPair(p);
      }
    }
    while (!readyPairs.empty()) {
      auto p = readyPairs.back();
      readyPairs.pop_back();
      if (containsEdge(p[1], p[0])) {
        std::ostringstream oss;
        oss << "Processing Pair (" << p[0] << "," << p[1] << ") but the Edge "
            << p[1] << "->" << p[0] << " is already present. "
            << "The Graph has contradictory alignment constraints.";
        throw error(oss.str());
      }
      if (!containsEdge(p[0], p[1])) {
        insertEdge(p[0], p[1]);
      }
      for (auto impl : implications[p]) {
        registerPair(impl);
      }
    }
  }

public:
  Graph(const Edges &edges_, const std::vector<std::array<OpId, 4>> &couples)
      : nOps(edges_.size()), haveBeenReady(nOps, false),
        nInsScheduled(nOps, 0), pairs(nOps) {

    log().debug("Initializing Graph");

    edgesForward.resize(nOps);
    edgesBackward.resize(nOps);

    for (OpId from = 0; from < nOps; ++from) {
      if (!edges_[from].empty()) {
        auto tos = util::unisorted(edges_[from]);
        edgesForward[from].reserve(tos.size());
        if (tos.back() >= nOps) {
          throw error("Invalid Edges in Graph constructor");
        }
        for (auto to : tos) {
          insertEdge(from, to);
        }
      }
    }

    constexpr Arrows keys{{{0, 1}, {1, 0}, {2, 3}, {3, 2}}};
    constexpr Arrows vals{{{2, 3}, {3, 2}, {0, 1}, {1, 0}}};

    for (const auto &x : couples) {
      for (uint64_t i = 0; i < 4; ++i) {
        auto k = keys[i];
        auto v = vals[i];
        Pair key{x[k[0]], x[k[1]]};
        Pair val{x[v[0]], x[v[1]]};
        implications[key].push_back(val);
        pairs[x[k[0]]].push_back(key);
      }
    }
  }

  bool containsEdge(OpId a, OpId b) const {
    return std::find(edgesForward[a].cbegin(), edgesForward[a].cend(), b) !=
           edgesForward[a].cend();
  }

  // Based on Kahn's algorithm.
  // To handle the coupled pairs:
  // Every time an Op is recognised as being ready for scheduling (no
  // unscheduled inputs), Pairs which involve it found, and their coupled
  // Pairs have constraints inserted. The coupled Pairs themselves are checked
  // for further coupled Pairs, and so all derived constraints are inserted.
  std::vector<OpId> getFiloSchedule() {

    log().debug("In Graph::getFiloSchedule");

    std::vector<OpId> schedule;
    schedule.reserve(nOps);
    for (OpId id = 0; id < nOps; ++id) {
      if (edgesBackward[id].size() == 0) {
        insertReady(id);
      }
    }

    while (!ready.empty()) {
      auto opId = ready.back();
      ready.pop_back();
      schedule.push_back(opId);
      for (auto idOut : edgesForward[opId]) {
        ++nInsScheduled[idOut];
        if (nInsScheduled[idOut] == edgesBackward[idOut].size()) {
          insertReady(idOut);
        }
      }
    }

    return schedule;
  }
};

std::vector<OpId>
getFiloSchedule(const Edges &fwdEdges,
                const std::vector<std::array<OpId, 4>> &couples) {
  Graph g(fwdEdges, couples);
  return g.getFiloSchedule();
}

} // namespace supercon
} // namespace schedule
} // namespace poprithms
