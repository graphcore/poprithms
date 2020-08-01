// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <numeric>
#include <unordered_map>

#include <poprithms/memory/alias/aliasusings.hpp>
#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/alias/nodes.hpp>
#include <poprithms/memory/alias/origins.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace alias {

TensorId Graph::concat(const std::vector<TensorId> &ids, uint64_t axis) {
  const auto arrShapes = getShapes(ids);
  auto outShape        = Shape::concat(arrShapes, axis);
  return createNode<Concat>(ids, outShape, axis);
}

TensorId Graph::allocate(const Shape &sh, Color color) {
  return createNode<Allocate>({}, sh, color);
}

TensorId Graph::reshape(TensorId id, const Shape &to) {
  return createNode<Reshape>({id}, to);
}

TensorId Graph::expand(TensorId id, const Shape &to) {
  return createNode<Expand>({id}, to);
}

TensorId Graph::reverse(TensorId id, const std::vector<uint64_t> &dims) {
  return createNode<Reverse>({id}, node(id).shape(), dims);
}

TensorId Graph::settsample(TensorId id, const Region &f) {
  return createNode<SettSample>({id}, {f.nelms()}, f);
}

TensorId Graph::dimshuffle(TensorId id, const Permutation &perm) {
  return createNode<Permute>({id}, perm.apply(node(id).shape().get()), perm);
}

template <class T, class... Args>
TensorId Graph::createNode(const std::vector<TensorId> &ins,
                           const Shape &shape,
                           Args... args) {
  TensorId id(nTensors());
  const Node::State ob(ins, {}, getShapes(ins), id, shape);
  nodes.push_back(UpNode(std::make_unique<T>(ob, Origins(shape), args...)));
  setOrigins(id.get());
  for (auto inId : node(id).ins()) {
    node(inId).insertOut(id);
  }
  return id;
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
  return *nodes[a.get()].up;
}

// See Scott Meyers' "Effective C++"
Node &Graph::node(TensorId id) {
  return const_cast<Node &>(static_cast<const Graph &>(*this).node(id));
}

std::vector<Shape>
Graph::getShapes(const std::vector<TensorId> &tenIds) const {
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

namespace {
template <typename T> std::string getStr(const std::vector<T> &X) {
  std::ostringstream ost;
  poprithms::util::append(ost, X);
  return ost.str();
}
} // namespace

void Graph::appendVerbose(std::ostream &oss) const {
  append(oss);
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
        oss << regs;
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
    shapes__[i + 2]      = getStr(node(i).shape().get());
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

  if (nd.allocates()) {
    nd.insertOrigin(AllocId(id.get()), {Region::createFull(shape(id))});
  }

  // `unwind' back to the allocations of the Node samples
  // (slice/subSample).
  //
  // Example:
  //
  // allocate(5,7) - dimshuffle({1,0}) - slice((1,2), (3,5))
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

  // For non-sampling Nodes, the origins are the same as the the input
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

bool Graph::isRowMajorSetContiguous(TensorId id) const {
  return node(id).isRowMajorSetContiguous();
}

bool Graph::containsAliases(TensorId id) const {
  return node(id).containsAliases();
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
std::vector<TensorId> Graph::allAliases(TensorId id) const {
  std::vector<TensorId> allAliased;

  std::vector<TensorId> toProcess{id};
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

std::vector<std::vector<TensorId>> Graph::allAliases() const {
  std::vector<std::vector<TensorId>> x(nTensors());
  for (uint64_t i = 0; i < nTensors(); ++i) {
    x[i] = allAliases(i);
  }
  return x;
}

template <typename F>
std::vector<TensorId> Graph::depthFirstBack(TensorId x0, F &&f) const {

  wspace.resize(nTensors());
  auto &currentEdge = wspace.wsUint64_;
  auto &scheduled   = wspace.wsBool_;

  if (!f(x0)) {
    return {};
  }
  std::vector<TensorId> sched;
  std::vector<uint64_t> S(1, x0.get());

  while (!S.empty()) {
    auto b = S.back();
    // All children explored, so can process this node (post-order
    // traversal).
    if (currentEdge[b] == node(b).ins().size()) {
      S.pop_back();
      sched.push_back(b);
      scheduled[b] = true;
    } else {
      auto to = node(b).ins()[currentEdge[b]];
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

std::vector<TensorId> Graph::depthFirstBackAll(TensorId x0) const {
  return depthFirstBack(x0, [](TensorId) { return true; });
}

std::vector<TensorId> Graph::depthFirstBackAliases(TensorId x0) const {
  return depthFirstBack(
      x0, [this, x0](TensorId id) { return id == x0 || areAliased(x0, id); });
}

void Graph::Workspace::clear(const std::vector<TensorId> &sched) {
  // clean-up
  for (auto x : sched) {
    wsBool_[x.get()]   = false;
    wsUint64_[x.get()] = 0;
  }
}

template <typename T>
Graph::Up<T>::Up(std::unique_ptr<T> x) : up(std::move(x)) {}

template <typename T>
bool Graph::Up<T>::operator==(const Graph::Up<T> &rhs) const {
  if ((up && !rhs.up) || (!up && rhs.up)) {
    return false;
  }
  if (!up && !rhs.up) {
    return true;
  }
  return (*up == *rhs.up);
}

template <typename T> Graph::Up<T>::Up() = default;

template <typename T>
Graph::Up<T>::Up(const Up &x) : Up<T>(x.up ? x.up->clone() : nullptr) {}

template <typename T> Graph::Up<T>::~Up() = default;

template <typename T>
Graph::Up<T> &Graph::Up<T>::operator=(const Graph::Up<T> &x) {
  if (x.up) {
    up = x.up->clone();
  }
  return *this;
}

// This implementation works vy performing a depth-first search for all
// Tensors preceding the Tensor clone, and creating a clone of all of them.
// Example:
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
TensorId Graph::clone(TensorId toCloneId) {
  const std::vector<TensorId> oldsToClone = depthFirstBackAll(toCloneId);

  auto &oldToNew = wspace.wsUint64_;

  for (uint64_t i = 0; i < oldsToClone.size(); ++i) {
    oldToNew[oldsToClone[i].get()] = nTensors() + i;
  }

  for (auto atc : oldsToClone) {
    const auto &toClone = node(atc);
    std::vector<TensorId> newIns;
    const TensorId newId(oldToNew[atc.get()]);

    for (auto oldIn : toClone.ins()) {
      newIns.push_back(oldToNew[oldIn.get()]);
    }
    Node::State newState(
        newIns, {}, toClone.inShapes(), newId, toClone.shape());

    auto newNode = toClone.clone(newState, toClone.origins().remap(oldToNew));

    for (auto inId : newIns) {
      node(inId).insertOut(newId);
    }
    nodes.push_back(std::move(newNode));
  }

  wspace.clear(oldsToClone);
  return nTensors() - 1;
}

std::string Graph::verboseString() const {
  std::ostringstream oss;
  appendVerbose(oss);
  return oss.str();
}

const Shape &Graph::shape(TensorId id) const { return node(id).shape(); }

// This must appear after template function implementations (above)
template class Graph::Up<Node>;

} // namespace alias
} // namespace memory
} // namespace poprithms