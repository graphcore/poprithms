// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <unordered_set>

#include <schedule/supercon/error.hpp>

#include <poprithms/schedule/supercon/graph.hpp>
#include <poprithms/schedule/supercon/logging.hpp>
#include <poprithms/util/unisort.hpp>

namespace poprithms {
namespace schedule {
namespace supercon {

namespace {
using Arrows = std::array<std::array<uint64_t, 2>, 4>;
using Pair   = std::array<NodeId, 2>;

static constexpr Arrows keys{{{0, 1}, {1, 0}, {2, 3}, {3, 2}}};
static constexpr Arrows vals{{{2, 3}, {3, 2}, {0, 1}, {1, 0}}};

class GraphImpl {
public:
  // number of Ops in the Graph
  uint64_t nOps;
  // The forward edges in the Graph
  Edges edgesForward;
  // The backward edges in the Graph
  Edges edgesBackward;

  // The Ops which have all their inputs scheduled, but have not themselves
  // yet been scheduled.
  std::vector<NodeId> ready;

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
  // temporarily stored in readyPairs (analogous to "ready" for Ops)
  std::vector<Pair> readyPairs;

  // A record of Pairs which are or have been in readyPairs is kept here:
  std::set<Pair> haveBeenReadyPairs;

  void insertEdge(NodeId from, NodeId to) {
    edgesForward[from].push_back(to);
    edgesBackward[to].push_back(from);
  }

  bool containsEdge(NodeId a, NodeId b) const {
    return std::find(edgesForward[a].cbegin(), edgesForward[a].cend(), b) !=
           edgesForward[a].cend();
  }

  void registerPair(Pair p) {
    if (haveBeenReadyPairs.count({p[0], p[1]}) == 0) {
      readyPairs.push_back(p);
      haveBeenReadyPairs.insert(p);
    }
  }

  void insertReady(NodeId id) {
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

  GraphImpl(const Edges &edges_, const Couples &couples)
      : nOps{edges_.size()}, edgesForward(edges_.size()),
        edgesBackward(edges_.size()), ready{},
        haveBeenReady(edges_.size(), false),
        nInsScheduled(edges_.size(), 0ull), pairs(edges_.size()),
        implications{}, readyPairs{}, haveBeenReadyPairs{} {

    edgesForward.resize(nOps);
    edgesBackward.resize(nOps);

    for (NodeId from = 0; from < nOps; ++from) {
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

  std::vector<NodeId> getFiloSchedule() {

    // Based on Kahn's algorithm.
    // To handle the coupled pairs:
    // Every time an Op is recognised as being ready for scheduling (no
    // unscheduled inputs), Pairs which involve it are found, and their
    // coupled Pairs have constraints inserted. The coupled Pairs themselves
    // are checked for further coupled Pairs, and so all derived constraints
    // are inserted.

    std::vector<NodeId> schedule;
    schedule.reserve(nOps);
    for (NodeId id = 0; id < nOps; ++id) {
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
} // namespace

Couple::Couple(const std::array<NodeId, 4> &rhs) : value{} {
  if ((rhs[0] == rhs[1]) || (rhs[2] == rhs[3]) ||
      ((rhs[0] == rhs[2]) && (rhs[1] == rhs[3])) ||
      ((rhs[0] == rhs[3]) && (rhs[1] == rhs[2]))) {
    std::stringstream ss;
    ss << "Invalid schedule::supercon::Couple ([";
    ss << rhs[0] << "," << rhs[1] << "," << rhs[2] << ",";
    ss << rhs[3] << "])";
    throw error(ss.str());
  } else {
    value = Couple::canonicalize(rhs);
  }
}

const NodeId &
Couple::operator[](std::array<NodeId, 4>::size_type index) const {
  return value[index];
}

bool Couple::operator<(const Couple &rhs) const { return value < rhs.value; }

bool Couple::operator==(const Couple &rhs) const {
  return value == rhs.value;
}

std::ostream &operator<<(std::ostream &out, const Couple &couple) {
  out << "[" << couple[0] << "," << couple[1] << ",";
  out << couple[2] << "," << couple[3] << "]";
  return out;
}

std::array<NodeId, 4>
Couple::canonicalize(const std::array<NodeId, 4> &value) {

  std::array<NodeId, 4> result = value;

  auto minl = std::min(result[0], result[1]);
  auto minr = std::min(result[2], result[3]);
  auto maxl = std::max(result[0], result[1]);
  auto maxr = std::max(result[2], result[3]);

  if (minl > minr || (minl == minr && maxl > maxr)) {
    // Need to swap pairs around.
    std::swap(result[0], result[2]);
    std::swap(result[1], result[3]);
    std::swap(minl, minr);
    std::swap(maxl, maxr);
  }
  if (result[0] > minl) {
    // Need to swap elements within pair around.
    std::swap(result[0], result[1]);
    std::swap(result[2], result[3]);
  }
  return result;
}

Graph::Graph() = default;

Graph::Graph(const Edges &edges_, const Couples &couples_)
    : edges(edges_.size()), couples() {

  log().debug("Initializing Graph");

  for (NodeId from = 0; from < nOps(); ++from) {
    for (auto to : edges_[from]) {
      insertEdge(from, to);
    }
  }

  for (const auto &x : couples_) {
    insertCouple(x);
  }
}

void Graph::insertEdge(NodeId from, NodeId to) {
  if (!containsEdge(from, to)) {
    grow(std::max(from, to));
    edges[from].push_back(to);
  }
}

bool Graph::containsEdge(NodeId a, NodeId b) const {
  if (a < nOps()) {
    return std::find(edges[a].cbegin(), edges[a].cend(), b) !=
           edges[a].cend();
  }
  return false;
}

void Graph::insertCouple(const Couple &couple) {
  if (!containsCouple(couple)) {
    for (size_t i = 0; i < 4; ++i)
      grow(couple[i]);
    couples.push_back(couple);
  }
}

bool Graph::containsCouple(const Couple &couple) const {
  return std::find(couples.cbegin(), couples.cend(), couple) !=
         couples.cend();
}

std::vector<NodeId> Graph::getFiloSchedule() const {
  GraphImpl impl{edges, couples};
  return impl.getFiloSchedule();
}

void Graph::grow(NodeId a) {
  if (nOps() <= a) {
    edges.resize(a + 1);
  }
}

uint64_t Graph::nOps() const { return edges.size(); }

std::vector<NodeId> getFiloSchedule(const Edges &fwdEdges,
                                    const Couples &couples) {
  Graph g(fwdEdges, couples);
  return g.getFiloSchedule();
}

} // namespace supercon
} // namespace schedule
} // namespace poprithms
