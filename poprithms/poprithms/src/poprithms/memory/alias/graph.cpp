// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>

#include <memory/alias/error.hpp>

#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/alias/nodes.hpp>
#include <poprithms/memory/alias/origins.hpp>
#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace alias {

TensorIds Graph::ids(const Tensors &ts) {
  TensorIds ids;
  ids.reserve(ts.size());
  std::transform(ts.cbegin(),
                 ts.cend(),
                 std::back_inserter(ids),
                 [](Tensor t) { return t.id(); });
  return ids;
}

Tensors Graph::tensors(const TensorIds &tIds) {
  Tensors ts;
  ts.reserve(tIds.size());
  for (const auto &id : tIds) {
    ts.push_back(tensor(id));
  }
  return ts;
}

Graph::~Graph() = default;

std::ostream &operator<<(std::ostream &ost, BroadcastPadding single) {
  switch (single) {
  case BroadcastPadding::Yes: {
    ost << "BroadcastPadding::Yes";
    break;
  }

  case BroadcastPadding::No: {
    ost << "BroadcastPadding::No";
    break;
  }
  }
  return ost;
}

TensorId Graph::concat(const TensorIds &ids, uint64_t axis) {
  const auto arrShapes = getShapes(ids);
  auto outShape        = Shape::concat(arrShapes, axis);
  return createNode<Concat>(ids, outShape, axis);
}

TensorId Graph::settfill(const TensorIds &ids,
                         const DisjointRegions &regions) {
  if (ids.size() == 0 || regions.size() == 0) {
    throw error(
        "settfill requires more than 0 inputs, and non-empty Regions");
  }
  const auto outShape = regions.at(0).shape();
  return createNode<SettFill>(ids, outShape, regions);
}

TensorId Graph::allocate(const Shape &sh, Color color) {
  return createNode<Allocate>({}, sh, color);
}

TensorId Graph::reshape(TensorId id, const Shape &to) {
  node(id).shape().assertSameNumberOfElements(to);
  return createNode<Reshape>({id}, to);
}

TensorId Graph::expand(TensorId id, const Shape &to) {
  node(id).shape().assertCanExpandTo(to);
  return createNode<Expand>({id}, to);
}

TensorId Graph::identity(TensorId id) {
  return createNode<Identity>({id}, node(id).shape());
}

void Graph::toAllocation(TensorId id, Color c) {

  const auto &n = node(id);

  // disconnect current inputs
  for (auto inId : n.ins()) {
    node(inId).removeOut(id);
  }

  const TensorIds newInputs{};
  completeInputlessReplacement<Allocate>(id, newInputs, c);
}

template <class T, class... Args>
void Graph::completeInputlessReplacement(TensorId beingTransformed,
                                         const TensorIds &newIns,
                                         Args... args) {

  const auto &n = node(beingTransformed);

  auto toUpdate = depthFirstFwdAliases(beingTransformed);
  std::reverse(toUpdate.begin(), toUpdate.end());

  nodes[beingTransformed.get()] = UpNode(createNodeWithOutsAndId<T>(
      newIns, n.outs(), n.shape(), n.id(), args...));

  for (auto idToUpdate : toUpdate) {
    setOrigins(idToUpdate);
  }
}

const TensorIds &Graph::ins(TensorId id) const { return node(id).ins(); }

const TensorIds &Graph::outs(TensorId id) const { return node(id).outs(); }

bool Graph::allocates(TensorId id) const { return node(id).allocates(); }

void Graph::allocationToConcat(const TensorIds &ins,
                               uint64_t axis,
                               TensorId allocId) {
  assertFromAllocation(allocId, Shape::concat(getShapes(ins), axis));
  completeInputlessReplacement<Concat>(allocId, ins, axis);
}

void Graph::assertFromAllocation(TensorId allocId,
                                 const Shape &expectedOut) const {
  const auto &n = node(allocId);
  if (n.nIns_i32() != 0) {
    std::ostringstream oss;
    oss << "Failed in assertFromAllocation(" << allocId << ", " << expectedOut
        << "). The number of inputs is not 0, but " << n.nIns_i32()
        << ", so the Tensor is not an allocation. ";
    throw error(oss.str());
  }

  if (expectedOut != n.shape()) {
    std::ostringstream oss;
    oss << "Failed in assertFromAllocation(" << allocId << ", " << expectedOut
        << "). The current output shape is " << n.shape()
        << ", shapes must match.";
    throw error(oss.str());
  }
}

void Graph::allocationToSettsample(TensorId inTensor,
                                   const Region &r,
                                   TensorId allocId) {
  assertFromAllocation(allocId, r.nelms());
  completeInputlessReplacement<SettSample>(allocId, {inTensor}, r);
}

void Graph::allocationToDimshuffle(TensorId inTensor,
                                   const Permutation &p,
                                   TensorId allocId) {
  assertFromAllocation(allocId, shape(inTensor).dimShuffle(p));
  completeInputlessReplacement<DimShuffle>(allocId, {inTensor}, p);
}

void Graph::allocationToReshape(TensorId inTensor, TensorId allocId) {
  shape(inTensor).assertSameNumberOfElements(shape(allocId));
  assertFromAllocation(allocId, shape(allocId));
  completeInputlessReplacement<Reshape>(allocId, {inTensor});
}

void Graph::allocationToExpand(TensorId inTensor, TensorId allocId) {
  shape(inTensor).assertCanExpandTo(shape(allocId));
  assertFromAllocation(allocId, shape(allocId));
  completeInputlessReplacement<Expand>(allocId, {inTensor});
}

void Graph::allocationToReverse(TensorId inTensor,
                                const std::vector<uint64_t> &dimensions,
                                TensorId allocId) {
  if (shape(inTensor) != shape(allocId)) {
    std::ostringstream oss;
    oss << "Failure in allocationToReverse, can only perform if "
        << "input and output shapes agree. " << shape(inTensor)
        << " != " << shape(allocId);
    throw error(oss.str());
  }
  assertFromAllocation(allocId, shape(allocId));
  completeInputlessReplacement<Reverse>(allocId, {inTensor}, dimensions);
}

void Graph::toIdentity(TensorId src, TensorId dst) {
  if (shape(src) != shape(dst)) {
    std::ostringstream oss;
    oss << "Incompatible Shapes in toIdentity: " << shape(src)
        << " != " << shape(dst) << ". Shapes must be identical. ";
    throw error(oss.str());
  }

  const auto &n = node(dst);

  // disconnect current inputs
  for (auto inId : n.ins()) {
    node(inId).removeOut(dst);
  }
  node(src).insertOut(dst);

  completeInputlessReplacement<Identity>(dst, {src});
}

std::vector<std::array<TensorId, 2>>
Graph::createBroadcastPadElements(const Shape &s,
                                  const std::vector<uint64_t> &l,
                                  const std::vector<uint64_t> &u,
                                  Color padColor) {
  const auto alloc     = allocate({}, padColor);
  const auto padShapes = s.getPadShapes(l, u);
  std::vector<std::array<TensorId, 2>> paddings;
  paddings.reserve(s.rank_u64());
  for (auto [l_, u_] : padShapes) {
    paddings.push_back({expand(alloc, l_), expand(alloc, u_)});
  }
  return paddings;
}

std::vector<std::array<TensorId, 2>>
Graph::createNonAliasedPadElements(const Shape &s,
                                   const std::vector<uint64_t> &l,
                                   const std::vector<uint64_t> &u,
                                   Color padColor) {
  std::vector<std::array<TensorId, 2>> paddings;
  paddings.reserve(s.rank_u64());
  for (auto [l_, u_] : s.getPadShapes(l, u)) {
    paddings.push_back({allocate(l_, padColor), allocate(u_, padColor)});
  }
  return paddings;
}

TensorId Graph::pad(TensorId id,
                    const std::vector<uint64_t> &l,
                    const std::vector<uint64_t> &u,
                    Color padColor,
                    BroadcastPadding singlePadElement) {
  auto paddingIds =
      (singlePadElement == BroadcastPadding::Yes)
          ? createBroadcastPadElements(shape(id), l, u, padColor)
          : createNonAliasedPadElements(shape(id), l, u, padColor);

  TensorId current = id;
  for (uint64_t d = 0; d < rank_u64(id); ++d) {
    current = concat(
        {std::get<0>(paddingIds[d]), current, std::get<1>(paddingIds[d])}, d);
  }
  return current;
}

TensorId Graph::reverse(TensorId id, const std::vector<uint64_t> &dims) {
  return createNode<Reverse>({id}, node(id).shape(), dims);
}

TensorId Graph::settSample(TensorId id, const Region &f) {
  return createNode<SettSample>({id}, {f.nelms()}, f);
}

TensorId Graph::dimShuffle(TensorId id, const Permutation &perm) {
  return createNode<DimShuffle>(
      {id}, perm.apply(node(id).shape().get()), perm);
}

template <class T, class... Args>
TensorId
Graph::createNode(const TensorIds &ins, const Shape &shape, Args... args) {
  TensorId id(nTensors());
  nodes.push_back(UpNode(
      createNodeWithOutsAndId<T, Args...>(ins, {}, shape, id, args...)));
  setOrigins(id.get());
  return id;
}

template <class T, class... Args>
std::unique_ptr<T> Graph::createNodeWithOutsAndId(const TensorIds &ins,
                                                  const TensorIds &outs,
                                                  const Shape &shape,
                                                  TensorId id,
                                                  Args... args) {
  const Node::State ob(ins, outs, getShapes(ins), id, shape);
  auto newNode = std::make_unique<T>(ob, Origins(shape), args...);
  for (auto inId : newNode->ins()) {
    node(inId).insertOut(id);
  }
  return newNode;
}

void Graph::Workspace::resize(uint64_t s) {
  wsBool_.resize(s, false);
  wsUint64_.resize(s, 0);
}

const Node &Graph::node(TensorId a) const {
  if (a.get() >= nTensors()) {
    std::ostringstream oss;
    oss << "The number of created Nodes is " << nTensors()
        << ", there is no Node with TensorId " << a;
    throw error(oss.str());
  }

  return *nodes[a.get()].uptr;
}

// See Scott Meyers' "Effective C++"
Node &Graph::node(TensorId id) {
  return const_cast<Node &>(static_cast<const Graph &>(*this).node(id));
}

std::vector<Shape> Graph::getShapes(const TensorIds &tenIds) const {
  std::vector<Shape> shapes;
  shapes.reserve(tenIds.size());
  for (const auto &id : tenIds) {
    shapes.push_back(node(id).shape());
  }
  return shapes;
}

std::ostream &operator<<(std::ostream &oss, const Graph &g) {
  g.append(oss);
  return oss;
}

void Graph::Workspace::reserve(uint64_t nArrs) {
  wsBool_.reserve(nArrs);
  wsUint64_.reserve(nArrs);
}

void Graph::appendOrigins(std::ostream &oss, bool bitwise) const {
  std::string spc(7, ' ');

  oss << "\n\n"
      << spc << "Origins:\n\n"
      << spc << "id  regions aliased in allocation Tensors"
      << "\n"
      << spc << "--  -------------------------------------";
  oss << '\n';
  for (uint64_t i = 0; i < nTensors(); ++i) {
    oss << "\n";
    int counter = 0;
    for (auto allocId : node(i).getAllocIds()) {
      if (counter == 0) {
        oss << "       " << i << ": ";
      } else {
        oss << "          ";
      }

      ++counter;
      oss << " [" << allocId << "]:(";
      for (const auto &regs : node(i).origins().at(allocId)) {
        if (!bitwise) {
          regs.append(oss);
        } else {
          regs.appendBitwise(oss);
        }
      }
      oss << ")\n";
    }
  }
}

void Graph::append(std::ostream &oss) const {

  // Note that "append" might be the slowest class method because of this
  // call, but I'm assuming that any user who wants to log the Graph isn't
  // too concerned.
  const auto aliasedTo = allAliases();

  // Return just enough white space to get a perfect alignment of columns.
  const auto getSpace = [](uint64_t target, const std::string &ts) {
    uint64_t taken = ts.size();
    if (taken > target) {
      return std::string(" ");
    }
    return std::string(target - taken + 1, ' ');
  };

  const auto nLines = nTensors() + 2;

  // Column titles
  std::vector<std::string> ids__(nLines, "id");
  std::vector<std::string> types__(nLines, "type");
  std::vector<std::string> ins__(nLines, "ins");
  std::vector<std::string> outs__(nLines, "outs");
  std::vector<std::string> shapes__(nLines, "shape");
  std::vector<std::string> selfAliases__(nLines, "aliases");
  std::vector<std::string> aliasedTo__(nLines, "aliased to");

  for (uint64_t i = 0; i < nTensors(); ++i) {
    ids__[i + 2]         = std::to_string(i);
    types__[i + 2]       = node(i).typeString();
    ins__[i + 2]         = getStr(node(i).ins());
    outs__[i + 2]        = getStr(node(i).outs());
    shapes__[i + 2]      = util::getStr(node(i).shape().get());
    std::string sa       = containsAliases(i) ? "yes" : "no";
    selfAliases__[i + 2] = sa;
    aliasedTo__[i + 2]   = getStr(aliasedTo[i]);
  }

  std::vector<std::vector<std::string>> frags{
      ids__, types__, ins__, shapes__, outs__, selfAliases__, aliasedTo__};

  auto getLen = [](const std::vector<std::string> &v) {
    return 1 + std::accumulate(v.cbegin(),
                               v.cend(),
                               0ULL,
                               [](size_t n, const std::string &x) {
                                 return std::max(n, x.size());
                               });
  };

  std::vector<uint64_t> lens;
  for (auto &f : frags) {
    const auto lw = getLen(f);
    lens.push_back(lw);
    f[1] = std::string(lw, '-');
  }

  for (uint64_t i = 0; i < nLines; ++i) {
    oss << "\n       ";
    for (uint64_t fi = 0; fi < frags.size(); ++fi) {
      oss << frags[fi][i] << getSpace(lens[fi], frags[fi][i]);
    }
  }
}

std::string Graph::typeString(TensorId id) const {
  return node(id).typeString();
}

namespace {
struct ToReverse {
public:
  ToReverse(TensorId i, DisjointRegions r) : id(i), regs(r) {}
  TensorId id;
  DisjointRegions regs;
};
} // namespace

void Graph::setOrigins(TensorId id) {

  auto &nd = node(id);
  nd.clearOrigins();

  if (shape(id).nelms_u64() == 0) {
    return;
  }

  if (nd.allocates()) {
    nd.insertOrigin(AllocId(id.get()), {Region::createFull(shape(id))});
  }

  // `unwind' back to the allocations of the Node samples
  // (slice/subSample).
  //
  // Example:
  //
  // allocate(5,7) - dimShuffle({1,0}) - slice((1,2), (3,5))
  // (5,7)         - (7,5)             - (2,3)
  //
  // what region in the allocation does the sliced Tensor of shape (2,3)
  // map to? Map it back through preceding layers.
  else if (nd.samples()) {

    // The Regions to trace back to their allocations.
    std::vector<ToReverse> toReverse{
        {id, DisjointRegions::createFull(nd.shape())}};

    while (!toReverse.empty()) {
      const auto current = toReverse.back();
      toReverse.pop_back();
      const auto &currentNode = node(current.id);
      if (currentNode.allocates()) {
        nd.insertOrigin(AllocId(current.id.get()), current.regs);
      } else {
        for (uint64_t ind0 = 0; ind0 < currentNode.ins().size(); ++ind0) {
          const auto regs = currentNode.getInRegions(ind0, current.regs);
          if (!regs.empty()) {
            toReverse.push_back({currentNode.in(ind0), regs});
          }
        }
      }
    }
  }

  // For non-sampling Nodes, the origins are the same as the input
  // Tensors. The input Tensor origins are guaranteed to by this point in
  // the code, as we are iterating through the Nodes in topological order.
  else {
    for (const auto inId : nd.ins()) {
      nd.insertOriginsFrom(node(inId));
    }
  }
}

bool Graph::areAliased(TensorId tenId0, TensorId tenId1) const {
  return node(tenId0).isAliasedTo(node(tenId1));
}

bool Graph::contains(TensorId super, TensorId sub) const {
  return node(super).contains(node(sub));
}

bool Graph::isRowMajorSetContiguous(TensorId id) const {
  return node(id).isRowMajorSetContiguous();
}

bool Graph::containsAliases(TensorId id) const {
  return node(id).containsAliases();
}

Colors Graph::colors(TensorId id) const {
  // Using set, not unordered_set, so the the returned vector is ordered.
  std::set<Color> colors;
  const auto allocIds = node(id).getAllocIds();
  for (auto allocId : allocIds) {
    const auto &allo       = node(allocId.get());
    const auto &asAllocate = dynamic_cast<const Allocate &>(allo);
    colors.insert(asAllocate.color());
  }
  return Colors(colors.cbegin(), colors.cend());
}

bool Graph::containsColor(TensorId id, Color c) const {
  const auto allocIds = node(id).getAllocIds();
  for (auto allocId : allocIds) {
    const auto &allo       = node(allocId.get());
    const auto &asAllocate = dynamic_cast<const Allocate &>(allo);
    if (asAllocate.color() == c) {
      return true;
    }
  }
  return false;
}

// edge case: what about the empty Tensor, does it alias itself? No, by
// definition of set intersection. A aliases B iff there exists at least 1
// element in both.
TensorIds Graph::allAliases(TensorId id) const {
  TensorIds allAliased;

  TensorIds toProcess{id};
  auto seen = toProcess;

  // perform breadth first search in both DAG directions.
  while (!toProcess.empty()) {
    const auto nxt = toProcess.back();
    toProcess.pop_back();
    if (areAliased(nxt, id)) {
      allAliased.push_back(nxt);
      for (auto o : node(nxt).insAndOuts()) {
        if (std::find(seen.cbegin(), seen.cend(), o) == seen.cend()) {
          toProcess.push_back(o);
          seen.push_back(o);
        }
      }
    }
  }

  std::sort(allAliased.begin(), allAliased.end());
  return allAliased;
}

std::vector<TensorIds> Graph::allAliases() const {
  std::vector<TensorIds> x(nTensors());
  for (uint64_t i = 0; i < nTensors(); ++i) {
    x[i] = allAliases(i);
  }
  return x;
}

std::map<TensorId, std::set<TensorId>> Graph::allAliasesMap() const {
  const auto v = allAliases();
  std::map<TensorId, std::set<TensorId>> m;
  for (uint64_t i = 0; i < v.size(); ++i) {
    std::set<TensorId> ids;
    for (auto id : v[i]) {
      ids.emplace(id);
    }
    m.insert({TensorId(i), ids});
  }
  return m;
}

void Graph::confirmAllAliasesMap(
    const std::map<TensorId, std::set<TensorId>> &m) const {
  const auto baseline = allAliasesMap();
  if (baseline == m) {
    return;
  }
  std::ostringstream oss;
  oss << "Different maps in Graph::confirmAllAliasesMap. ";

  // keys which are not in baseline:
  for (const auto &[k, s] : m) {
    (void)s;
    const auto found = baseline.find(k);
    if (found == baseline.cend()) {
      oss << "\n    --> No key " << k << " in baseline.";
    }
  }

  for (const auto &[k, s] : baseline) {
    const auto found = m.find(k);

    // keys not in target:
    if (found == m.cend()) {
      oss << "\n    --> No key " << k << " in target.";
    } else {
      if (found->second != s) {
        oss << "\n    --> For key " << k << " the baseline has ";
        util::append(oss, TensorIds(s.cbegin(), s.cend()));
        oss << " and the target has ";
        util::append(oss,
                     TensorIds(found->second.cbegin(), found->second.cend()));
        oss << ".";
      }
    }
  }
  throw error(oss.str());
}

template <Graph::Direction D> const TensorIds &next(const Node &n);

template <> const TensorIds &next<Graph::Direction::Fwd>(const Node &n) {
  return n.outs();
}

template <> const TensorIds &next<Graph::Direction::Bwd>(const Node &n) {
  return n.ins();
}

template <Graph::Direction D, class F>
TensorIds Graph::depthFirst(TensorId x0, F &&f) const {

  wspace.resize(nTensors());
  auto &currentEdge = wspace.wsUint64_;
  auto &scheduled   = wspace.wsBool_;

  if (!f(x0)) {
    return {};
  }
  TensorIds sched;
  std::vector<uint64_t> S(1, x0.get());

  while (!S.empty()) {
    auto b = S.back();
    // All children explored, so can process this node (post-order
    // traversal).
    if (currentEdge[b] == next<D>(node(b)).size()) {
      S.pop_back();
      sched.push_back(b);
      scheduled[b] = true;
    } else {
      auto to = next<D>(node(b))[currentEdge[b]];
      if (!scheduled[to.get()] && f(to)) {
        S.push_back(to.get());
      }
      ++currentEdge[b];
    }
  }

  // clean up the workspace for the next call. Note for multi-threading:
  // will need as many workspaces as threads.
  wspace.clear(sched);
  return sched;
}

template <typename F>
TensorIds Graph::depthFirstBwd(TensorId x0, F &&f) const {
  return depthFirst<Direction::Bwd>(x0, f);
}

template <typename F>
TensorIds Graph::depthFirstFwd(TensorId x0, F &&f) const {
  return depthFirst<Direction::Fwd>(x0, f);
}

TensorIds Graph::depthFirstBwdAll(TensorId x0) const {
  return depthFirstBwd(x0, [](TensorId) { return true; });
}

TensorIds Graph::depthFirstBwdAliases(TensorId x0) const {
  return depthFirstBwd(
      x0, [this, x0](TensorId id) { return id == x0 || areAliased(x0, id); });
}

TensorIds Graph::depthFirstFwdAliases(TensorId x0) const {
  return depthFirstFwd(
      x0, [this, x0](TensorId id) { return id == x0 || areAliased(x0, id); });
}

void Graph::Workspace::clear(const TensorIds &sched) {
  // clean-up
  for (auto x : sched) {
    wsBool_[x.get()]   = false;
    wsUint64_[x.get()] = 0;
  }
}

// This implementation works by performing a depth-first search for all
// Tensors preceding the Tensor being cloned, and creating a clone of all of
// them. Example:
//
//     allocate - reverse -|- concat - slice (toClone)
//     allocate - permute -|
//
//  The depth-first post-order traversal returns
//  (allocate, reverse, allocate, permute, concat, slice).
//
//  This method then creates a clone of each. This ensures that the
//  allocations of each of the cloned Nodes mirrors the original sub-tree's
//  corresponding Node.
//
//  TODO(jn) It could be more efficient in some cases. For example, if
//  slice.containsAlias(permute), the path back from permute could be
//  pruned. Note that replacing depthFirstBackAll with depthFirstBackAliases
//  is not enough, as that would leave concat with a "dangling" input.
//
TensorId Graph::clone(TensorId toCloneId, CloneColorMethod cloneColorMethod) {
  const TensorIds oldsToClone = depthFirstBwdAll(toCloneId);

  auto &oldToNew = wspace.wsUint64_;

  for (uint64_t i = 0; i < oldsToClone.size(); ++i) {
    oldToNew[oldsToClone[i].get()] = nTensors() + i;
  }

  for (auto atc : oldsToClone) {
    const auto &toClone = node(atc);
    TensorIds newIns;
    const TensorId newId(oldToNew[atc.get()]);

    for (auto oldIn : toClone.ins()) {
      newIns.push_back(oldToNew[oldIn.get()]);
    }
    Node::State newState(
        newIns, {}, toClone.inShapes(), newId, toClone.shape());

    auto newNode = [&toClone, &newState, &oldToNew, cloneColorMethod]() {
      const auto remappedOrigins = toClone.origins().remap(oldToNew);

      // For nodes which do not allocate:
      if (!toClone.allocates()) {
        return toClone.clone(newState, remappedOrigins);
      }

      // For Allocate nodes:
      // We check if the clone is Monochrome or Preserving (in poplar,
      // Monochrome corresponds to always non-constant).
      if (auto asAllocate = dynamic_cast<const Allocate *>(&toClone)) {

        const auto cloneColor = cloneColorMethod.isMonochrome()
                                    ? cloneColorMethod.monochromeColor()
                                    : asAllocate->color();

        return asAllocate->cloneWithColor(
            newState, remappedOrigins, cloneColor);
      }

      // A node which allocates but is not an Allocate node? Something must
      // have changed, is there are new kind of node which allocates?
      std::ostringstream oss;
      oss << "The node " << toClone.str()
          << " allocates, but it not of type Allocate. "
          << "This needs to be handled in Graph::clone.";
      throw error(oss.str());
    }();

    for (auto inId : newIns) {
      node(inId).insertOut(newId);
    }
    nodes.push_back(std::move(newNode));
  }

  wspace.clear(oldsToClone);
  return nTensors() - 1;
}

const std::vector<DisjointRegions> &
Graph::allocRegions(TensorId query, TensorId allocation) const {
  if (!allocates(allocation)) {
    error("Not an allocation");
  }

  return node(query).origins().at(AllocId(allocation.get()));
}

std::string Graph::verboseString() const {
  std::ostringstream oss;
  append(oss);
  appendSettwiseOrigins(oss);
  return oss.str();
}

void Graph::reserve(uint64_t nTensors) { nodes.reserve(nTensors); }

const Shape &Graph::shape(TensorId id) const { return node(id).shape(); }

bool Graph::operator==(const Graph &rhs) const { return nodes == rhs.nodes; }

} // namespace alias
} // namespace memory

namespace util {
template class CopyByClone<memory::alias::Node>;
}
} // namespace poprithms
