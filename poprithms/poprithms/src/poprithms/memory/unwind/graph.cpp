// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <memory/unwind/ops.hpp>
#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/memory/unwind/solution.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>
#include <poprithms/util/unisort.hpp>
#include <util/copybyclone_impl.hpp>

// For now in this translation unit
#include <poprithms/memory/unwind/solution.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

SubGraphId Graph::subGraphId(const TensorId &id) const {
  return op(id.opId()).subGraphId();
}

SubGraphIds Graph::subGraphIds(const TensorIds &tids) const {
  SubGraphIds ids;
  ids.reserve(tids.size());
  for (auto tid : tids) {
    ids.push_back(subGraphId(tid));
  }
  return ids;
}

TensorIds Graph::call(SubGraphId outer,
                      SubGraphId inner,
                      const TensorIds &inSources,
                      const TensorIds &inDests,
                      const TensorIds &outSources,
                      double v) {

  return call(outer,
              inner,
              inSources,
              inDests,
              outSources,
              std::vector<double>(inSources.size(), v),
              std::vector<double>(outSources.size(), v));
}

TensorIds Graph::call(SubGraphId outer,
                      SubGraphId inner,
                      const TensorIds &inSources,
                      const TensorIds &inDests,
                      const TensorIds &outSources,
                      const std::vector<double> &copyInValues,
                      const std::vector<double> &copyOutValues) {

  if (copyInValues.size() != inSources.size() ||
      copyInValues.size() != inDests.size()) {
    std::ostringstream oss;
    oss << "In call, copyInValues is of size " << copyInValues.size()
        << ", inSources is of size " << inSources.size()
        << ", and inDests is of size " << inDests.size()
        << ". They should all be of the same size. ";
    throw error(oss.str());
  }

  if (copyOutValues.size() != outSources.size()) {
    std::ostringstream oss;
    oss << "In call, copyOutValues is of size " << copyOutValues.size()
        << ", and outSources is of size " << outSources.size()
        << ". They should be the same. ";
  }

  auto assertSubGraphId = [this, outer, inner](bool isOuter,
                                               const TensorIds &ids) {
    const auto target = isOuter ? outer : inner;
    for (const auto &id : ids) {
      if (subGraphId(id) != target) {
        std::ostringstream oss;
        oss << "Failure in assertSubGraphId, the Tensor " << id
            << " is not in the expected subGraph. ";
        throw error(oss.str());
      }
    }
  };

  assertSubGraphId(true, inSources);
  assertSubGraphId(false, inDests);
  assertSubGraphId(false, outSources);

  TensorIds outs;
  for (uint64_t i = 0; i < outSources.size(); ++i) {
    outs.push_back(sink(shape(outSources[i]), outer));
  }

  for (uint64_t i = 0; i < inSources.size(); ++i) {
    insertValuedPair(inSources[i], inDests[i], copyInValues[i]);
  }

  for (uint64_t i = 0; i < outSources.size(); ++i) {
    insertValuedPair(outSources[i], outs[i], copyOutValues[i]);
  }

  return outs;
}

bool Graph::isSink(const TensorId &id) const {
  return op(id.opId()).isSink(id.outIndex());
}

bool Graph::isSource(const TensorId &id) const {
  return op(id.opId()).isSource(id.outIndex());
}

DisjointRegions Graph::outRegions(const DisjointRegions &inRegions,
                                  InIndex inIndex,
                                  OpId opId,
                                  OutIndex outIndex) const {
  return op(opId).outRegions(inRegions, inIndex, outIndex);
}

DisjointRegions Graph::inRegions(const DisjointRegions &out,
                                 InIndex inIndex,
                                 OpId opId,
                                 OutIndex outIndex) const {
  return op(opId).inRegions(out, inIndex, outIndex);
}

SubGraphId Graph::subGraphIdFromTensorIds(const TensorIds &ids) const {
  if (ids.empty()) {
    std::ostringstream oss;
    oss << "Failed to obtain SubGraphId from empty vector of TensorIds. ";
    throw error(oss.str());
  }

  const auto sgid = subGraphId(ids[0]);
  if (std::any_of(
          ids.cbegin() + 1, ids.cend(), [&sgid, this](const auto &id) {
            return subGraphId(id) != sgid;
          })) {
    std::ostringstream oss;
    oss << "Contradictory solutions in obtaining "
        << "SubGraphId from TensorIds, " << ids
        << ". Expected all TensorIds to have same SubGraphId, "
        << "but the SubGraphIds are " << subGraphIds(ids);
    throw error(oss.str());
  }

  return sgid;
}

template <class T, class... Args>
OpId Graph::createOpWithInputs(const TensorIds &inIds,
                               const Shapes &outShapes,
                               Args... args) {

  return insertOp(
      std::make_unique<T>(Op::getStartingState(nOps_i64(),
                                               subGraphIdFromTensorIds(inIds),
                                               inIds,
                                               shapes(inIds),
                                               outShapes),
                          args...));
}

template <class T, class... Args>
OpId Graph::createInputlessOp(SubGraphId sgid,
                              const Shapes &outShapes,
                              Args... args) {

  return insertOp(std::make_unique<T>(
      Op::getStartingState(nOps_i64(), sgid, {}, {}, outShapes), args...));
}

void Graph::insertValuedPair(const TensorId &a, const TensorId &b, double v) {
  if (shape(a) != shape(b)) {
    std::ostringstream oss;
    oss << "Invalid valuedPair pair, (" << a << ", " << b
        << "). Tensors in valuedPair pairs must have same Shape, but "
        << shape(a) << " != " << shape(b) << '.';
    throw error(oss.str());
  }
  op(a.opId()).insertAttractor(a.outIndex(), b, v);
  op(b.opId()).insertAttractor(b.outIndex(), a, v);
}

OpId Graph::insertOp(std::unique_ptr<Op> createdOp) {
  const auto newId = insertMultioutOp(std::move(createdOp));
  return newId;
}

SumLikeOut
Graph::sumLike(const TensorIds &ids, InIndex unwindIndex, double val) {

  if (ids.size() <= unwindIndex.get()) {
    std::ostringstream oss;
    oss << "Graph::sumLike(ids = " << ids << ", unwindIndex = " << unwindIndex
        << ") is invalid. "
        << "unwindIndex should be less than " << ids.size()
        << ", the number of inputs. ";
    throw error(oss.str());
  }

  const TensorId outId{
      createOpWithInputs<SumLike>(
          ids, {Shape::numpyVariadic(shapes(ids))}, unwindIndex),
      0};

  SumLikeMappings mappings;

  // for every pair of inputs:
  for (uint64_t iA = 0; iA < ids.size(); ++iA) {
    for (uint64_t iB = iA + 1; iB < ids.size(); ++iB) {

      const auto idA = ids[iA];
      const auto idB = ids[iB];
      const auto shA = shape(idA);
      const auto shB = shape(idB);

      const auto insertUniDir = [this, val, &mappings, &iA, &iB](
                                    const TensorId &from,
                                    const TensorId &to) {
        const auto layoutSrc = sumLikeReduce(from, shape(to));
        if (!getName(from.opId()).empty()) {
          setName(layoutSrc.opId(),
                  "sumLike-reduce(" + getName(from.opId()) + "(" +
                      std::to_string(iA) + "->" + std::to_string(iB) + "))");
        }
        insertValuedPair(layoutSrc, to, val);
        mappings.push_back({from, layoutSrc.opId(), to});
      };

      // insert valuedPair if shapes same.
      if (shA == shB) {
        insertValuedPair(idA, idB, val);
      }

      // else if A's shape dominates B's shape:
      else if (shA.numpyBinary(shB) == shA) {
        insertUniDir(idA, idB);
      }

      // else if B's shape dominates A's shape:
      else if (shB.numpyBinary(shA) == shB) {
        insertUniDir(idB, idA);
      }
    }
  }

  return {outId, mappings};
}

TensorId Graph::settSample(const TensorId &id, const Region &r) {
  return {createOpWithInputs<SettSample>({id}, {r.nelms()}, r), 0};
}
TensorId Graph::dimShuffle(const TensorId &id, const Permutation &perm) {
  return {createOpWithInputs<DimShuffle>(
              {id}, {shape(id).dimShuffle(perm)}, perm),
          0};
}
TensorId Graph::reverse(const TensorId &id, const Dimensions &d) {
  return {createOpWithInputs<Reverse>({id}, {shape(id)}, d), 0};
}

TensorId Graph::identity(const TensorId &id) {
  return {createOpWithInputs<Identity>({id}, {shape(id)}), 0};
}
TensorId Graph::reshape(const TensorId &id, const Shape &outShape) {
  return {createOpWithInputs<Reshape>({id}, {outShape}), 0};
}

bool Graph::isUnwindable(OpId opId,
                         InIndex inIndex,
                         OutIndex outIndex) const {
  return op(opId).isUnwindable(inIndex, outIndex);
}

void Graph::extendBwd(Chain &c,
                      OpId opId,
                      InIndex inIndex,
                      OutIndex outIndex) const {
  op(opId).extendBwd(c, inIndex, outIndex);
}

bool Graph::isBarrier(const TensorId &tId) const {
  return op(tId.opId()).isBarrier(tId.outIndex());
}

void Graph::extend(Chain &c, const Link &l) const {
  op(l.opId()).extend(c, l.inIndex(), l.outIndex(), l.isFwd());
}

void Graph::extend(Chain &c, const Links &ls) const {
  for (const auto &l : ls) {
    extend(c, l);
  }
}

TensorId Graph::concat(const TensorIds &ids, uint64_t axis) {
  const auto shapes_ = shapes(ids);
  Shape::assertConcattable(shapes_, axis);
  return {
      createOpWithInputs<Concat>(ids, {Shape::concat(shapes_, axis)}, axis),
      0};
}

TensorIds Graph::sinks() const {
  TensorIds ids;
  for (uint64_t i = 0; i < nOps(); ++i) {
    const auto &op_ = op(i);
    for (uint64_t o = 0; o < op_.nOutTensors(); ++o) {
      if (op(i).isSink(o)) {
        ids.push_back({i, o});
      }
    }
  }
  return ids;
}

namespace {
template <typename T> TensorIds all(const Graph &g, const T &t) {
  TensorIds ids;
  for (auto tId : g.tensorIds()) {
    if (t(tId)) {
      ids.push_back(tId);
    }
  }
  return ids;
}
} // namespace

TensorIds Graph::sources() const {
  return all(*this, [this](const TensorId &id) {
    return op(id.opId()).isSource(id.outIndex());
  });
}

TensorIds Graph::barriers() const {
  return all(*this, [this](const TensorId &id) {
    return op(id.opId()).isBarrier(id.outIndex());
  });
}

TensorIds Graph::sourcesAndBarriers() const {
  return all(*this, [this](const TensorId &id) {
    return op(id.opId()).isBarrier(id.outIndex()) ||
           op(id.opId()).isSource(id.outIndex());
  });
}

TensorId Graph::sink(const Shape &shape, SubGraphId sgid) {
  return {createInputlessOp<Sink>(sgid, {shape}, sgid), 0};
}
TensorId
Graph::sink(const Shape &shape, SubGraphId sgid, const std::string &name) {
  auto id = sink(shape, sgid);
  setName(id.opId(), name);
  return id;
}

Chain Graph::extended(const Chain &c,
                      InIndex inIndex,
                      OpId opId,
                      OutIndex outIndex) const {
  auto c_ = c;

  extend(c_, Link(opId, inIndex, outIndex, true));
  return c_;
}

std::string Graph::name(SubGraphId id) const {
  const auto found = sgNames.find(id);
  if (found == sgNames.cend()) {
    return "";
  }
  return found->second;
}

Path Graph::fullEmpty(const TensorId &src, const TensorId &dst) const {
  if (shape(src) != shape(dst)) {
    std::ostringstream oss;
    oss << "Error in Graph::fullEmpty(src=" << src << ", dst=" << dst
        << "). The Shape of src is " << shape(src)
        << ", and the Shape of dst is " << shape(dst)
        << ". They should be the same. ";
  }
  return Path(src, chain::Chain(shape(src)), dst);
}

Path Graph::extendedPath(const Path &p,
                         InIndex inIndex,
                         OpId opId,
                         OutIndex outIndex) const {
  const auto extendedChain = extended(p.chain(), inIndex, opId, outIndex);
  return Path(p.src(), extendedChain, {opId, outIndex});
}

TensorId Graph::source(const Shape &shape, SubGraphId sgid) {
  return {createInputlessOp<Source>(sgid, {shape}, sgid), 0};
}
TensorId
Graph::source(const Shape &shape, SubGraphId sgid, const std::string &name) {
  auto id = source(shape, sgid);
  setName(id.opId(), name);
  return id;
}

OpId Graph::barrier(const TensorIds &inIds, const Shapes &outShapes) {
  if (inIds.empty()) {
    throw error("Graph::barrier({}) invalid. At least 1 input required.");
  }
  return createOpWithInputs<Barrier>(inIds, outShapes);
}

TensorId Graph::sumLikeReduce(const TensorId &id, const Shape &out) {
  return {createOpWithInputs<SumLikeReduce>({id}, {out}), 0};
}

const Op &Graph::op(OpId a) const {
  // We know that all Ops in this Graph can be safely cast, do not need a
  // dynamic_cast here.
  return static_cast<const Op &>(multioutOp(a));
}
Op &Graph::op(OpId a) { return static_cast<Op &>(multioutOp(a)); }

namespace {
template <typename T> std::string getStr(const std::vector<T> &X) {
  std::ostringstream ost;
  poprithms::util::append(ost, X);
  return ost.str();
}
} // namespace

void Graph::append(std::ostream &ost) const {

  const auto nLines = nTensors() + 5;

  using Strings = std::vector<std::string>;
  Strings opId__(nLines, "");
  opId__[0] = "OpId";

  Strings opDebugString__(nLines, "");
  opDebugString__[0] = "Name";

  Strings opType__(nLines, "");
  opType__[0] = "OpType";

  Strings inTensors__(nLines, "");
  inTensors__[0] = "InTensors";

  Strings outIndex__(nLines, "");
  outIndex__[0] = "OutIndex";

  Strings tensorShape__(nLines, "");
  tensorShape__[0] = "Shape";

  // extensions:

  Strings copyAttractors__(nLines, "");
  copyAttractors__[0] = "Attractors";

  uint64_t l = 2;
  for (uint64_t i = 0; i < nOps(); ++i) {

    opId__[l]          = std::to_string(i);
    opType__[l]        = op(i).typeString();
    opDebugString__[l] = op(i).getName();
    inTensors__[l]     = getStr((op(i).inTensorIds()));
    // ++l;
    for (uint64_t o = 0; o < op(i).nOutTensors(); ++o) {
      outIndex__[l]       = std::to_string(o);
      tensorShape__[l]    = getStr(shape({i, o}).get());
      copyAttractors__[l] = getStr(op(i).valuedPartners());
      ++l;
    }
    if (op(i).nOutTensors() == 0) {
      ++l;
    }
  }

  std::vector<Strings> frags;
  frags.push_back(opId__);

  // only add debug strings if at least one has:
  if (std::any_of(opDebugString__.cbegin() + 2,
                  opDebugString__.cend(),
                  [](const auto &x) { return !x.empty(); })) {
    frags.push_back(opDebugString__);
  }

  frags.push_back(opType__);
  frags.push_back(inTensors__);

  if (std::any_of(
          outIndex__.cbegin() + 2, outIndex__.cend(), [](const auto &x) {
            return (!x.empty() && x[0] != '0');
          })) {
    frags.push_back(outIndex__);
  }

  frags.push_back(tensorShape__);

  frags.push_back(copyAttractors__);

  const auto getLen = [](const Strings &v) {
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
    ost << "\n       ";
    for (uint64_t fi = 0; fi < frags.size(); ++fi) {
      ost << frags[fi][i] << util::spaceString(lens[fi], frags[fi][i]);
    }
  }
}

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

ValuedTensorIds Graph::valuedPartners(const TensorId &tId) const {
  return op(tId.opId()).valuedPartners(tId.outIndex());
}

ValuedPairs Graph::valuedPairs() const {
  ValuedPairs allAttractors;
  for (auto tId : tensorIds()) {
    for (auto att : valuedPartners(tId)) {
      if (att.tensorId() <= tId) {
        allAttractors.push_back({tId, att.tensorId(), att.value()});
      }
    }
  }
  return allAttractors;
}

TensorId Graph::slice(const TensorId &id, const Lower &l, const Upper &u) {
  return settSample(id, Region::fromBounds(shape(id), l, u));
}

TensorId
Graph::slice(const TensorId &id, Dimension d, uint64_t l, uint64_t u) {
  return settSample(id, Region::fromBounds(shape(id), d, l, u));
}

TensorId Graph::subSample(const TensorId &id, Stride s, Dimension d) {
  return settSample(id, Region::fromStride(shape(id), s, d));
}

TensorId Graph::subSample(const TensorId &id, const Strides &strides) {
  return settSample(id, Region::fromStrides(shape(id), strides));
}

TensorId Graph::flatten(const TensorId &id) {
  return reshape(id, {shape(id).nelms()});
}

Path Graph::getPath(const TensorId &src,
                    const Links &links,
                    const TensorId &dst) const {
  Chain chain(shape(src));
  extend(chain, links);
  return Path(src, chain, dst);
}

TensorId Graph::inTensorId(InIndex inIndex, OpId opId) const {
  return op(opId.get()).inTensorId(inIndex);
}

TensorId Graph::squeeze(const TensorId &id) {
  return reshape(id, shape(id).squeeze());
}

TensorId Graph::slice(const TensorId &id, uint64_t l, uint64_t u) {
  return slice(id, Dimension(0), l, u);
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
