// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <schedule/shift/graphserialization.hpp>

#include <boost/functional/hash.hpp>

#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/filteredschedule.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/logging.hpp>
#include <poprithms/schedule/shift/schedulechange.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>
#include <poprithms/util/unisort.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

OpAddress Graph::insertOp(const std::string &dbs) {
  OpAddress op = nOps();
  allOps.push_back({op, dbs});
  return op;
}

std::vector<OpAddress>
Graph::insertOps(const std::vector<std::string> &dbStrings) {
  std::vector<OpAddress> opAdds;
  opAdds.reserve(dbStrings.size());
  for (const auto &dbs : dbStrings) {
    opAdds.push_back(insertOp(dbs));
  }
  return opAdds;
}

std::vector<std::array<OpAddress, 2>> Graph::getTightPairs() const {

  std::vector<std::array<OpAddress, 2>> tightPairs;
  for (const auto &op : getOps()) {
    if (op.nOuts() == 1UL && getOp(op.getOut(0)).nIns() == 1UL) {
      tightPairs.push_back(
          {op.getAddress(), getOp(op.getOut(0)).getAddress()});
    }
  }
  return tightPairs;
}

std::vector<OpAddress> Graph::tightChainFrom(OpAddress a) const {
  std::vector<OpAddress> chain;
  chain.push_back(a);
  auto crr = a;
  while (getOp(crr).nOuts() == 1 && getOp(getOp(crr).getOut(0)).nIns() == 1) {
    chain.push_back(getOp(crr).getOut(0));
    crr = getOp(crr).getOut(0);
  }
  return chain;
}

std::vector<std::vector<OpAddress>> Graph::getTightChains() const {
  std::vector<std::vector<OpAddress>> tightChains;

  constexpr OpAddress None{std::numeric_limits<OpAddress>::max()};

  std::vector<OpAddress> tightNext(nOps(), None);
  std::vector<OpAddress> tightPrev(nOps(), None);
  for (const auto &op : getOps()) {
    if (op.nOuts() == 1 && getOp(op.getOut(0)).nIns() == 1) {
      tightNext[op.getAddress()] = op.getOut(0);
      tightPrev[op.getOut(0)]    = op.getAddress();
    }
  }

  for (OpAddress add = 0; add < nOps(); ++add) {
    if (tightNext[add] != None && tightPrev[add] == None) {
      auto chainElm = add;
      tightChains.push_back({chainElm, tightNext[chainElm]});
      chainElm = tightNext[chainElm];
      while (tightNext[chainElm] != None) {
        chainElm = tightNext[chainElm];
        tightChains.back().push_back(chainElm);
      }
    }
  }

  return tightChains;
}

void Graph::insertOpAlloc(OpAddress oa, AllocAddress aa) {
  allAllocs[aa].insertOp(oa);
  allOps[oa].insertAlloc(aa);
}

void Graph::insertOpAlloc(const std::vector<OpAddress> &oas,
                          AllocAddress aa) {
  for (auto oa : oas) {
    insertOpAlloc(oa, aa);
  }
}

void Graph::insertBinConstraints(
    const std::vector<std::vector<OpAddress>> &bins,
    const std::string &prefix) {
  OpAddress prev = insertOp(prefix + std::to_string(0));
  for (uint64_t i = 1; i < bins.size(); ++i) {
    // a "bottleneck" Op, which partitions Ops into different bins.
    auto op = insertOp(prefix + std::to_string(i));
    for (auto b : bins[i - 1]) {
      insertConstraint(b, op);
    }
    for (auto a : bins[i]) {
      insertConstraint(op, a);
    }
    insertConstraint(prev, op);
    prev = op;
  }
}

void Graph::insertAttractions(
    const std::vector<std::array<OpAddress, 2>> &knots,
    AllocWeight w) {
  for (const auto &knot : knots) {
    auto allocAddress = insertAlloc(w);
    insertOpAlloc(std::get<0>(knot), allocAddress);
    insertOpAlloc(std::get<1>(knot), allocAddress);
  }
}

void Graph::removeConstraint(OpAddress before, OpAddress after) {
  allOps[before].removeOut(after);
  allOps[after].removeIn(before);
}

void Graph::insertConstraint(OpAddress before, OpAddress after) {
  if (before >= nOps() || after >= nOps()) {
    std::ostringstream oss;
    oss << "Cannot insert constraint " << before << " -> " << after
        << ", as there are only " << nOps() << " Ops in the shift::Graph";
    throw error(oss.str());
  }
  if (!allOps[before].hasOut(after)) {
    allOps[before].insertOut(after);
    allOps[after].insertIn(before);
  }
}

void Graph::insertLink(OpAddress before, OpAddress after) {
  if (!getOp(before).hasOut(after)) {
    insertConstraint(before, after);
  }

  const auto &op0 = getOp(before);
  const auto &op1 = getOp(after);

  if (op0.hasForwardLink() && op0.getForwardLink() != after) {
    std::ostringstream oss;
    oss << "Ops can have at most one link forward. "
        << "Op " << op0 << " already has " << getOp(op0.getForwardLink())
        << " as a forward link, and so " << op1
        << " cannot be added as a forward link.";
    throw error(oss.str());
  }

  if (op1.hasBackwardLink() && op1.getBackwardLink() != before) {
    std::ostringstream oss;
    oss << "Ops can have at most one link backward. "
        << "Op " << op1 << " already has " << getOp(op1.getBackwardLink())
        << " as a backward link, and so " << op0
        << " cannot be added as a backward link.";
    throw error(oss.str());
  }

  allOps[before].insertForwardLink(after);
  allOps[after].insertBackwardLink(before);
}

void Graph::insertConstraints(
    const std::vector<std::array<OpAddress, 2>> &cs) {
  for (const auto &c : cs) {
    insertConstraint(std::get<0>(c), std::get<1>(c));
  }
}

void Graph::append(std::ostream &ost) const {
  for (auto op : getOps()) {
    ost << '\n' << op.getDebugString() << "   <-  [";
    for (auto inAdd : op.getIns()) {
      ost << ' ' << getOp(inAdd).getDebugString() << ' ';
    }
    ost << ']';
  }
}

AllocAddress Graph::insertAlloc(AllocWeight w) {
  AllocAddress a = allAllocs.size();
  allAllocs.push_back({a, w});
  return a;
}

Graph::OpMerged Graph::getLinkMerged() const {
  return getMerged(getLinkChains());
}

Graph::OpMerged Graph::getTightMerged() const {
  return getMerged(getTightChains());
}

Graph::OpMerged
Graph::getMerged(const std::vector<std::vector<OpAddress>> &chains) const {

  Graph childGraph;

  // The Allocs are the same in the child Graph as the parent Graph
  for (const auto &parentAlloc : getAllocs()) {
    childGraph.insertAlloc(parentAlloc.getWeight());
  }

  constexpr OpAddress None{std::numeric_limits<OpAddress>::max()};

  // Map an Op in the parent Graph to its unique Op in the child Graph
  std::vector<OpAddress> parentToChild(nOps(), None);

  // We assign lowest addresses to child Ops which are generated from parent
  // chains, then the remaining addresses are assigned to the unchained Ops
  uint64_t childOpAddress{0};
  while (childOpAddress < chains.size()) {
    for (const auto opAddress : chains[childOpAddress]) {
      parentToChild[opAddress] = childOpAddress;
    }
    ++childOpAddress;
  }

  // Map an Op in the child Graph to its parent(s) in the parent Graph
  ParentGraphOps childToParents = std::move(chains);

  for (uint64_t parentAddress = 0; parentAddress < nOps(); ++parentAddress) {
    if (parentToChild[parentAddress] == None) {
      parentToChild[parentAddress] = childOpAddress;
      ++childOpAddress;
      childToParents.push_back({parentAddress});
    }
  }

  const auto nChildOps = childOpAddress;

  for (uint64_t childAddress = 0; childAddress < nChildOps; ++childAddress) {
    // The child Op's name is a concatenation of the names of the parent Ops
    const auto &parentAddresses = childToParents[childAddress];
    std::ostringstream ossChildName;
    ossChildName << '(';
    for (uint64_t i = 0; i < parentAddresses.size(); ++i) {
      if (i != 0) {
        ossChildName << ' ';
      }
      ossChildName << getOp(parentAddresses[i]).getDebugString();
    }
    ossChildName << ')';
    auto name = ossChildName.str();
    childGraph.insertOp(name);
  }

  for (uint64_t childAddress = 0; childAddress < nChildOps; ++childAddress) {

    // child Op inherits constraints and Allocs from parent(s)
    for (auto parentAddress : childToParents[childAddress]) {
      const auto &parent = getOp(parentAddress);
      for (auto allocAddress : parent.getAllocs()) {
        childGraph.insertOpAlloc(childAddress, allocAddress);
      }
      for (auto outParentAddress : parent.getOuts()) {
        auto outChildAddress = parentToChild[outParentAddress];
        if (outChildAddress != childAddress) {
          childGraph.insertConstraint(childAddress, outChildAddress);
        }
      }
    }
  }

  return {childGraph, childToParents};
}

std::vector<OpAddress> Graph::getOpsWithFwdLinks() const {
  std::vector<OpAddress> ops;
  for (const auto &op : getOps()) {
    if (op.hasForwardLink()) {
      ops.push_back(op.getAddress());
    }
  }
  return ops;
}

std::vector<std::vector<OpAddress>> Graph::getLinkChains() const {

  std::vector<std::vector<OpAddress>> chains;

  for (auto address : getOpsWithFwdLinks()) {
    // start of a chain
    if (!getOp(address).hasBackwardLink()) {
      chains.push_back({});
      auto current = address;
      while (getOp(current).hasForwardLink()) {
        chains.back().push_back(current);
        current = getOp(current).getForwardLink();
      }
      chains.back().push_back(current);
    }
  }

  return chains;
}

std::vector<std::vector<uint64_t>> Graph::getFwdEdges_u64() const {
  std::vector<std::vector<uint64_t>> edges(nOps());
  for (uint64_t from = 0; from < nOps(); ++from) {
    edges[from] = getOp(from).getOuts();
  }
  return edges;
}

std::vector<std::vector<OpAddress>> Graph::getForwardEdges() const {
  std::vector<std::vector<OpAddress>> fwdEdges(nOps());
  for (const auto &op : getOps()) {
    fwdEdges[op.getAddress()] = op.getOuts();
  }
  return fwdEdges;
}

std::vector<OpAddress> Graph::getIdenticalIns(OpAddress a) const {
  std::vector<OpAddress> sameIns;
  const auto &ins = getOp(a).getIns();
  if (ins.empty()) {
    return getInputOps();
  }
  auto in0 = ins[0];
  for (auto out : getOp(in0).getOuts()) {
    if (getOp(out).getIns() == ins) {
      sameIns.push_back(out);
    }
  }
  return sameIns;
}

uint64_t Graph::nConstraints() const {
  return std::accumulate(
      allOps.cbegin(), allOps.cend(), 0ULL, [](uint64_t n, const Op &op) {
        return n + op.nIns();
      });
}

std::ostream &operator<<(std::ostream &ost, const Graph &x) {
  x.append(ost);
  return ost;
}

void Graph::appendSerialization(std::ostream &ost) const {

  constexpr const char *const newline = "\n     ";
  ost << "{\"ops\":[";
  if (nOps() != 0) {
    ost << newline;
    getOp(0).appendSerialization(ost);
  }
  for (uint64_t i = 1; i < nOps(); ++i) {
    ost << ',';
    ost << newline;
    getOp(i).appendSerialization(ost);
  }
  ost << "],\n\"allocs\":[";
  if (nAllocs() != 0) {
    ost << newline;
    getAlloc(0).appendSerialization(ost);
  }
  for (uint64_t i = 1; i < nAllocs(); ++i) {
    ost << ',';
    ost << newline;
    getAlloc(i).appendSerialization(ost);
  }
  ost << "]}";
}

Graph Graph::fromSerializationString(const std::string &s) {
  return serialization::fromSerializationString(s);
}

void Graph::insertStartAttractorsAssert0(uint64_t opAddsSize,
                                         uint64_t priosSize) const {
  if (opAddsSize != priosSize) {
    std::ostringstream oss;
    oss << "Number of opAddresses (" << opAddsSize
        << ") is not the same as the number of priorities (" << priosSize
        << ") in insertStartAttractors.";
    throw error(oss.str());
  }
}

std::string Graph::getSerializationString() const {
  std::ostringstream oss;
  appendSerialization(oss);
  return oss.str();
}

std::vector<std::vector<OpAddress>>
Graph::constraintDiff(const std::vector<std::vector<OpAddress>> &rhs) const {

  if (nOps() != rhs.size()) {
    std::ostringstream oss;
    oss << "Graph::constraintDiff can only be called on "
        << "Edges with the same number of Ops as thisGraph. "
        << "This Graph has " << nOps() << " but \"rhs\" has " << rhs.size();
    throw error(oss.str());
  }

  std::vector<std::vector<OpAddress>> uniqueToThis(nOps());
  for (OpAddress x0 = 0; x0 < nOps(); ++x0) {
    for (OpAddress x1 : getOp(x0).getOuts()) {
      if (std::find(rhs[x0].cbegin(), rhs[x0].cend(), x1) == rhs[x0].cend()) {
        uniqueToThis[x0].push_back(x1);
      }
    }
    uniqueToThis[x0] = util::unisorted(uniqueToThis[x0]);
  }
  return uniqueToThis;
}

std::vector<OpAddress> Graph::getInputOps() const {
  std::vector<OpAddress> inputs;
  for (const auto &op : allOps) {
    if (op.nIns() == 0) {
      inputs.push_back(op.getAddress());
    }
  }
  return inputs;
}

bool Graph::operator==(const Graph &rhs) const {
  // include Op names in the comparison:
  return equalTo(rhs, true);
}
bool Graph::operator!=(const Graph &rhs) const { return !operator==(rhs); }
bool Graph::operator<(const Graph &rhs) const {
  // include Op names in the comparison
  return lessThan(rhs, true);
}
bool Graph::operator<=(const Graph &rhs) const { return !operator>(rhs); }
bool Graph::operator>(const Graph &rhs) const {
  return !operator==(rhs) && !operator<(rhs);
}
bool Graph::operator>=(const Graph &rhs) const { return !operator<(rhs); }

bool Graph::equalTo(const Graph &rhs, bool includeNames) const {
  if (allAllocs != rhs.allAllocs) {
    return false;
  }
  if (includeNames) {
    return allOps == rhs.allOps;
  } else {
    if (allOps.size() != rhs.allOps.size()) {
      return false;
    }
    for (uint64_t i = 0; i < allOps.size(); ++i) {
      if (getOp(i).getGraphComparator() !=
          rhs.getOp(i).getGraphComparator()) {
        return false;
      }
    }
  }
  return true;
}

bool Graph::lessThan(const Graph &rhs, bool includeNames) const {
  if (includeNames) {
    const auto a = std::tuple{getOps(), getAllocs()};
    const auto b = std::tuple{rhs.getOps(), rhs.getAllocs()};
    return a < b;
  }

  else {
    std::vector<Op::GraphComparator> as;
    as.reserve(nOps());
    for (const auto &op : getOps()) {
      as.push_back(op.getGraphComparator());
    }
    const auto A = std::tuple{as, getAllocs()};

    std::vector<Op::GraphComparator> bs;
    bs.reserve(rhs.nOps());
    for (const auto &op : rhs.getOps()) {
      bs.push_back(op.getGraphComparator());
    }
    const auto B = std::tuple{bs, rhs.getAllocs()};

    return A < B;
  }
}

size_t Graph::hash(bool includeNames) const {

  size_t hash = 0u;

  for (const auto &op : getOps()) {
    boost::hash_combine(hash, op.hash(includeNames));
  }

  for (const auto &alloc : getAllocs()) {
    boost::hash_combine(hash, alloc.hash());
  }

  return hash;
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
