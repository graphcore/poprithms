// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
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

#include <memory/unwind/error.hpp>
#include <memory/unwind/ops.hpp>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>
#include <poprithms/util/unisort.hpp>

// For now in this translation unit
#include <poprithms/memory/unwind/solution.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

TensorIds Graph::call(const TensorIds &inSources,
                      const TensorIds &inDests,
                      const TensorIds &outSources,
                      double v) {

  return call(inSources,
              inDests,
              outSources,
              std::vector<double>(inSources.size(), v),
              std::vector<double>(outSources.size(), v));
}

TensorIds Graph::call(const TensorIds &inSources,
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

  TensorIds outs;
  for (uint64_t i = 0; i < outSources.size(); ++i) {
    outs.push_back(sink(shape(outSources[i])));
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

template <class T, class... Args>
OpId Graph::createOp(const TensorIds &inIds,
                     const Shapes &outShapes,
                     Args... args) {

  return insertOp(std::make_unique<T>(
      Op::getStartingState(nOps_i64(), inIds, outShapes, *this), args...));
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

SumLikeOut Graph::sumLike(const TensorIds &ids,
                          const std::vector<InIndex> &uwIs,
                          const SumAttractions &attractions) {

  /**
   * Ops which are not unwindable through any indices are barriers.
   * */
  if (uwIs.empty()) {
    const TensorId outId{
        createOp<Barrier>(ids, {Shape::numpyVariadic(shapes(ids))}), 0};
    return {outId, SumLikeMappings()};
  }

  for (auto unwindIndex : uwIs) {
    if (ids.size() <= unwindIndex.get()) {
      std::ostringstream oss;
      oss << "Graph::sumLike(ids = " << ids
          << ", an unwindIndex = " << unwindIndex << ") is invalid. "
          << "The unwind indices should be less than " << ids.size()
          << ", the number of inputs. ";
      throw error(oss.str());
    }
  }

  const Shape outShape = Shape::numpyVariadic(shapes(ids));

  const TensorId outId{createOp<SumLike>(ids, {outShape}, uwIs), 0};

  SumLikeMappings mappings;

  // for every pair of inputs:
  for (uint64_t iA = 0; iA < ids.size(); ++iA) {
    for (uint64_t iB = iA + 1; iB < ids.size(); ++iB) {

      const auto idA = ids[iA];
      const auto idB = ids[iB];
      if (idA != idB) {
        const auto shA = shape(idA);
        const auto shB = shape(idB);

        const auto insertUniDir =
            [&attractions, this, &mappings, &iA, &iB, &ids](
                InIndex fromIndex_, InIndex toIndex_) {
              auto from                = ids[fromIndex_.get()];
              auto to                  = ids[toIndex_.get()];
              const TensorId layoutSrc = sumLikeReduce(from, shape(to));
              if (!getName(from.opId()).empty()) {
                setName(layoutSrc.opId(),
                        "sumLike-reduce(" + getName(from.opId()) +
                            "(InIndex:" + std::to_string(fromIndex_.get()) +
                            "->" + std::to_string(toIndex_.get()) + "))");
              }
              insertValuedPair(layoutSrc, to, attractions.get(iA, iB));
              mappings.push_back({from, layoutSrc.opId(), to});
            };

        // insert valuedPair if shapes same.
        if (shA == shB) {
          insertValuedPair(idA, idB, attractions.get(iA, iB));
        }

        // else if A's shape dominates B's shape:
        else if (shA.numpyBinary(shB) == shA) {
          insertUniDir(iA, iB);
        }

        // else if B's shape dominates A's shape:
        else if (shB.numpyBinary(shA) == shB) {
          insertUniDir(iB, iA);
        }
      }
    }
  }

  return {outId, mappings};
}

TensorId Graph::settSample(const TensorId &id, const Region &r) {
  return {createOp<SettSample>({id}, {r.nelms()}, r), 0};
}
TensorId Graph::dimShuffle(const TensorId &id, const Permutation &perm) {
  return {createOp<DimShuffle>({id}, {shape(id).dimShuffle(perm)}, perm), 0};
}
TensorId Graph::reverse(const TensorId &id, const Dimensions &d) {
  return {createOp<Reverse>({id}, {shape(id)}, d), 0};
}

TensorId Graph::identity(const TensorId &id) {
  return {createOp<Identity>({id}, {shape(id)}), 0};
}
TensorId Graph::reshape(const TensorId &id, const Shape &outShape) {
  return {createOp<Reshape>({id}, {outShape}), 0};
}

TensorId Graph::expand(const TensorId &id, const Shape &outShape) {
  return {createOp<Expand>({id}, {outShape}), 0};
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
  return {createOp<Concat>(ids, {Shape::concat(shapes_, axis)}, axis), 0};
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
    return op(id.opId()).isBarrier(id.outIndex()) &&
           op(id.opId()).nInTensors() == 0;
  });
}

TensorIds Graph::barriers() const {
  return all(*this, [this](const TensorId &id) {
    return op(id.opId()).isBarrier(id.outIndex());
  });
}

TensorId Graph::sink(const Shape &shape, const std::string &n) {
  const auto opId = createOp<Sink>({}, {shape});
  setName(opId, n);
  return {opId, 0};
}

Chain Graph::extended(const Chain &c,
                      InIndex inIndex,
                      OpId opId,
                      OutIndex outIndex) const {
  auto c_ = c;

  extend(c_, Link(opId, inIndex, outIndex, true));
  return c_;
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

OpId Graph::barrier(const TensorIds &inIds,
                    const Shapes &outShapes,
                    const std::string &n) {
  const auto opId = createOp<Barrier>(inIds, outShapes);
  setName(opId, n);
  return opId;
}

TensorId Graph::sumLikeReduce(const TensorId &full, const Shape &reduced) {
  return {createOp<SumLikeReduce>({full}, {reduced}), 0};
}

bool Graph::isSumLikeReduce(OpId opId) const {
  return dynamic_cast<const SumLikeReduce *>(&op(opId)) != nullptr;
}

const Op &Graph::op(OpId a) const {
  // We know that all Ops in this Graph can be safely cast, do not need a
  // dynamic_cast here.
  return static_cast<const Op &>(multioutOp(a));
}
Op &Graph::op(OpId a) { return static_cast<Op &>(multioutOp(a)); }

void Graph::appendOpColumns(std::ostream &ost, const OpIds &opIds) const {

  auto cols = getMultioutColumns(opIds, {});
  std::vector<std::string> copyAttractors(nMultioutRows(), "");

  uint64_t ti = 0;
  for (auto opId : opIds) {
    const auto &op_ = op(opId);
    for (uint64_t o = 0; o < op_.nOutTensors(); ++o) {
      copyAttractors[ti] = util::getStr(op_.valuedPartners(o));
      ++ti;
    }
    if (op_.nOutTensors() == 0) {
      ++ti;
    }
  }

  cols.push_back({"Attractors", copyAttractors, {}});
  ost << alignedColumns(cols);
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

TensorId Graph::squeeze(const TensorId &id) {
  return reshape(id, shape(id).squeeze());
}

TensorId Graph::slice(const TensorId &id, uint64_t l, uint64_t u) {
  return slice(id, Dimension(0), l, u);
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
