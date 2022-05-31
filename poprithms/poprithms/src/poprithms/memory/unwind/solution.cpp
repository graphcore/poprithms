// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

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
#include <poprithms/memory/unwind/solution.hpp>
#include <poprithms/schedule/vanilla/pathcount.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>
#include <poprithms/util/unisort.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

DisjointRegions Solution::coveredByPaths(const TensorId &tId) const {
  std::vector<Region> regions;
  for (const auto &p : inwardsPaths(tId)) {
    const auto &rs = p.dstRegions().get();
    regions.insert(regions.end(), rs.cbegin(), rs.cend());
  }
  auto cbp = DisjointRegions(graph().shape(tId), regions);
  return cbp;
}

//
bool Solution::completelyCoveredByPaths(const TensorId &tId) const {
  const auto rs = coveredByPaths(tId);
  return rs.totalElms() == graph().nelms(tId);
}
//
bool Solution::completelyCoveredByPaths(OpId id) const {
  for (OutIndex o = 0; o < graph().nOutTensors(id); ++o) {
    if (!completelyCoveredByPaths({id, o})) {
      return false;
    }
  }
  return true;
}

void Solution::clearInwardsPaths(OpId id) {
  inwardsPaths_[id.get()] = std::vector<Paths>(graph().nOutTensors(id));
}

void Solution::clearPathsBackToSinks(OpId id) {
  pathsBackToSinks_[id.get()] = std::vector<Paths>(graph().nOutTensors(id));
}

bool Solution::completelyCoveredByPaths(const TensorIds &ids) const {

  for (auto tId : ids) {
    if (!completelyCoveredByPaths(tId)) {
      return false;
    }
  }
  return true;
}

const Paths &Solution::pathsBackToSinks(const TensorId &src) const {
  return pathsBackToSinks_[src.opId().get()][src.outIndex().get()];
}

const Paths &Solution::inwardsPaths(const TensorId &id) const {
  return inwardsPaths_[id.opId().get()][id.outIndex().get()];
}

bool Solution::completelyCoveredByPaths() const {
  return completelyCoveredByPaths(graph().tensorIds());
}

void Solution::resetAllPathInfo() {
  for (uint64_t i = 0; i < graph().nOps(); ++i) {
    clearInwardsPaths(OpId(i));
  }
  pathStack.clear();
  setPathsBackToSinks();
}

void Solution::insertPath(const Path &p) {
  auto c0 = p.chain();
  c0.canonicalize();
  insertInwardsPath(p.dst(), {p.src(), c0, p.dst()});
}

void Solution::setPathsBackToSinks() {

  for (auto opId : graph().opIds()) {
    clearPathsBackToSinks(opId);
  }

  for (auto sink : graph().sinks()) {

    const Path p0(sink, chain::Chain(graph().shape(sink)), sink);
    insertPathBackToSink(sink, p0);

    Paths pathFront{p0};
    while (!pathFront.empty()) {
      const auto p = pathFront.back();
      pathFront.pop_back();
      for (auto c : graph().consumptionIds(p.src())) {
        const auto inIndex = c.inIndex();
        const auto opId    = c.opId();
        for (uint64_t outIndex = 0; outIndex < graph().nOutTensors(opId);
             ++outIndex) {
          if (graph().isUnwindable(opId, inIndex, outIndex)) {
            auto extendedChain =
                chain::Chain(graph().shape({opId, outIndex}));
            graph().extendBwd(extendedChain, opId, inIndex, outIndex);
            extendedChain.append(p.chain());
            const Path np({opId, outIndex}, extendedChain, p.dst());
            insertPathBackToSink({opId, outIndex}, np);
            pathFront.push_back(np);
          }
        }
      }
    }
  }
}

double Solution::getScore() const {

  double score{0.0};
  // double maxScore{0.0};
  const auto allAttractors = graph().valuedPairs();
  for (const auto &att : allAttractors) {
    // maxScore += att.valPerElm() * (graph().nelms(att.id0()));
    const auto &paths0 = inwardsPaths(att.id0());
    const auto &paths1 = inwardsPaths(att.id1());
    for (const auto &p0 : paths0) {
      for (const auto &p1 : paths1) {
        if (p0.src() == p1.src()) {
          const auto allInter = p0.dstRegions().intersect(p1.dstRegions());
          for (auto dstIntersection : allInter.get()) {
            const auto dstIntersectionSize = dstIntersection.totalElms();

            if (dstIntersectionSize != 0) {
              auto c0 = p0.chain();
              c0.settSample(dstIntersection);
              c0.canonicalize();

              auto c1 = p1.chain();
              c1.settSample(dstIntersection);
              c1.canonicalize();

              if (c0 == c1) {
                score += att.valPerElm() * dstIntersectionSize;
              }
            }
          }
        }
      }
    }
  }
  return score;
}

void Solution::setPathStackToSources() {
  // Sources have no dependencies, their layouts are know immediately.
  for (const auto src : graph().sources()) {
    // Sources define layouts, so their Paths destinations are the just the
    // Tensors (source) themselves.
    const auto dst = src;
    const auto p   = graph().fullEmpty(src, dst);
    insertPath(p);
    pathStack.push_back(p);
  }
}

Solution::Solution(const Graph &g, Algo a) : Solution(Graph(g), a) {}

Solution::Solution(Graph &&g, Algo a) : graph_(std::move(g)) {

  initPaths();

  switch (a) {
  case (Algo::Greedy0): {
    setPathsGreedy0();
    break;
  }
  default: {
    throw error("unrecognised Unwinding Algorithm");
  }
  }

  assertCompletelyCovererdByPaths();
}

void Solution::assertCompletelyCovererdByPaths() const {

  if (!completelyCoveredByPaths()) {
    TensorIds notCovered;
    for (auto tId : graph().tensorIds()) {
      if (!completelyCoveredByPaths(tId)) {
        notCovered.push_back(tId);
      }
    }
    std::ostringstream oss;
    oss << "Failed in assertCompletelyCovererdByPaths. "
        << "The Graph may have been underspecified. "
        << "All Tensor Sources must be included, including fall backs. "
        << "(map linearly in Poplar for example). "
        << "The tensors which are not completely covered are: " << notCovered;
    throw error(oss.str());
  }
}

void Solution::initPaths() {
  for (auto opId : graph().opIds()) {
    inwardsPaths_.push_back(std::vector<Paths>(graph().nOutTensors(opId)));
    pathsBackToSinks_.push_back(
        std::vector<Paths>(graph().nOutTensors(opId)));
  }
}

Solution::Solution(const Graph &g, const Paths &sourcesAndBsToSinks)
    : graph_(g) {
  initPaths();
  setAllPaths(sourcesAndBsToSinks);
  assertCompletelyCovererdByPaths();
}

void Solution::setPathsGreedy0() {

  auto fwdEdgeMap = graph().getMultioutForwardEdgeMap_u64();

  // The longest path, for each op, to a terminal op.
  auto lengths = [&fwdEdgeMap]() {
    using namespace poprithms::schedule::vanilla;
    return PathCounter::count(fwdEdgeMap.fwdEdgesCompact(),
                              CountType::Max,
                              ErrorIfCycle::No,
                              VerifyEdges::No);
  }();

  resetAllPathInfo();
  setPathStackToSources();

  // TODO(T33922) the next step is refinement. starting from a complete
  // solution, see which values were not obtained and see if an adjustment
  // can make overall score increase.

  std::priority_queue<ExtendedValuedPair> valueQueue;

  bool madeProgress{true};

  while (madeProgress) {

    madeProgress = false;
    if (!pathStack.empty()) {
      madeProgress        = true;
      const auto extended = processPathStack();

      // For all the Tensors which had some part of their layouts determined
      // during processing the stack:
      for (auto source : extended) {

        // For all Tensors which have an attraction to the newly determined
        // Tensor:
        for (auto destination : graph().valuedPartners(source)) {

          // Emplace a potential pair for unwinding: from the #source to
          // the #destination. The pair is weighted by its attraction
          // value, and the distance from the destination to a terminal
          // node is used as a tie-breaker (the logic behind this tie-breaker
          // is that tensors appearing 'early' in the compute graph
          // should have their layouts set earlier).
          ExtendedValuedPair p{
              source,
              destination.tensorId(),
              destination.value(),
              lengths[fwdEdgeMap.compactId(destination.tensorId().opId())]};
          valueQueue.push(p);
        }
      }
    }

    // keep looking for a new unwind path as long as the next element of
    // valueQueue doesn't provide one.
    bool keepLookingForUnwind{true};
    while (keepLookingForUnwind && !valueQueue.empty()) {

      // [t0]    The Source or Barrier Tensor. This is where the unwinding
      //  |      Path begins. Note that the direction of the unwinding Path is
      //  |       arbitrary, it can go forwards and backwards in the compute
      //  |        DAG.
      //  |
      //  +-+
      //    |
      //  +-+
      //  |
      // [t1]    This is the source side of the value pair. We'll try and map
      //  |      its known layout to a sink somewhere (t3), so that value
      //  |       points from the value pair can be obtained.
      //  v
      //  +----+
      //       |
      //    filters1   In creating a new Path from the Source/Barrier t0 to
      //       |       the Sink t3, we must ensure not to create any
      //       |      overlapping Paths; each element of each Tensor must
      //       |     correspond to exactly 1 Source/Barrier element.
      //       |
      //  +----+
      //  |
      //  |
      // [t2]   This is the sink side of the value pair. We want to set its
      //  |      layout to match t1's, or at least some new parts should
      //  |       match. Because when t1 and t2 match, points are obtained.
      //  |
      //  v
      //  |
      //  |
      // [t3]  This is the sink Tensor, which we ultimately must create a
      //         path to starting from t0, the source Tensor.
      //

      auto nxt = valueQueue.top();
      valueQueue.pop();

      const auto t1       = nxt.id0();
      const auto t2       = nxt.id1();
      const auto &paths01 = inwardsPaths(t1);
      const auto filters1 = coveredByPaths(t2).getComplement();

      for (const auto &path01 : paths01) {

        for (const auto &filter1 : filters1.get()) {

          const auto t0 = path01.src();
          auto chain02  = path01.chain();
          chain02.mask(filter1);

          for (const auto &path23 : pathsBackToSinks(t2)) {

            const auto t3 = path23.dst();
            auto chain03_ = chain02;
            chain03_.append(path23.chain());
            const auto filters = coveredByPaths(t3).getComplement().get();

            for (const auto &filter3 : filters) {
              auto chain03 = chain03_;
              chain03.mask(filter3);
              chain03.canonicalize();
              const auto path03 = Path(t0, chain03, t3);
              if (!path03.dstRegions().empty()) {
                madeProgress         = true;
                keepLookingForUnwind = false;
                insertPath(path03);
                pathStack.push_back(path03);
                barriersToSinks_.push_back(path03);
              }
            }
          }
        }
      }
    }
  }
}

TensorIds Solution::processPathStack() {

  TensorIds extended;

  auto add = [this](const Path &p) {
    insertPath(p);
    pathStack.push_back(p);
  };

  // While there is a Path whose destination might be unwindable forward,
  // process it
  while (!pathStack.empty()) {

    const auto currPath = pathStack.back();
    extended.push_back(currPath.dst());
    pathStack.pop_back();

    for (const auto c : graph().consumptionIds(currPath.dst())) {

      // The index of the consumer where the Tensor currPath.dst() is
      // consumed:
      const auto inInd = c.inIndex();

      // const auto &op_ = op(c.opId());
      const auto opId = c.opId();

      // Check for unwindability to each of the consumers outputs:
      for (auto outId : graph().outTensorIds(opId)) {

        const auto outInd = outId.outIndex();

        // If the path through the consumer is a barrier, use barrier
        // semantics: you cannot unwind through a barrier, but the output
        // can have a new empty Path initialized if the barrier Op has all
        // of its inputs layouts completely known:
        if (graph().isBarrier({opId, outInd})) {
          if (completelyCoveredByPaths(graph().inTensorIds(opId)) &&
              !completelyCoveredByPaths(outId)) {
            add(graph().fullEmpty(outId, outId));
          }
        }

        // not a barrier, so unwindable.
        else if (graph().isUnwindable(opId, inInd, outInd)) {

          auto outRegs =
              graph().outRegions(currPath.dstRegions(), inInd, opId, outInd);

          // This is required only in the case where unwinding is not 1:1 such
          // as when you have sums which are unwindable along multiple axes,
          // with repeated inputs. For example, out=a+a.
          outRegs = outRegs.intersect(
              coveredByPaths({opId, outInd}).getComplement());

          if (!outRegs.empty()) {
            add(graph().extendedPath(currPath, inInd, opId, outInd));
          }
        }
      }
    }
  }

  return util::unisorted(extended);
}

void Solution::processUnwindPath(const Path &tail) {

  //
  //   t0  a source Tensor
  //   |
  //   |
  //   +--+
  //      |
  //      |
  //      t1  the middle tensor, which is start of tail
  //      |
  //      |
  //   +--+
  //   |
  //   |
  //   t2  the sink Tensor, which is the end of tail
  //

  const auto t2 = tail.dst();
  if (!graph().isSink(t2)) {
    throw error("Solution Path destinations must be sinks");
  }
  if (!tail.dstRegions().disjoint(coveredByPaths(t2))) {
    throw error(
        "Target of Solution Path intersects already mapped Region in sink");
  }

  // The tail provided is the tail (from "mid" to "sink") of
  // the overall Path, which should come from a source (why should it?).
  // Unpack tail:
  const auto &t1      = tail.src();
  const auto &chain12 = tail.chain();

  const auto regs21 = chain12.mirror().apply(tail.dstRegions());
  if (!coveredByPaths(t1).contains(regs21)) {
    std::ostringstream oss;
    oss << "Error in processUnwindPath. The tail Path provided is\n"
        << tail << ". "
        << "If the tail Paths chain is mirrored (reversed), "
        << "and applied to tail's destination Tensor, "
        << "the Region in " << tail.src() << " of\n"
        << regs21 << " is obtained. "
        << " The problem is that only \n"
        << coveredByPaths(t1) << "is currently covered by Paths. "
        << "So at this point in the code, "
        << "we're processing an unwind Path where "
        << "some of the required input does not have a layout. ";
    throw error(oss.str());
  }

  for (const Path &path01 : inwardsPaths(t1)) {
    const auto t0 = path01.src();
    auto chain02  = path01.chain();
    chain02.append(chain12);
    const auto p = Path(t0, chain02, t2);
    insertPath(p);
    pathStack.push_back(p);
  }
}

void Solution::setAllPaths(const Paths &barriersToSinks) {
  resetAllPathInfo();
  setPathStackToSources();
  for (const auto &unwindSolnPath : barriersToSinks) {
    processPathStack();
    processUnwindPath(unwindSolnPath);
  }
  processPathStack();
  assertCompletelyCovererdByPaths();
}

void Solution::insertInwardsPath(const TensorId &id, const Path &p) {
  const auto pathDest = p.dst();
  if (id != pathDest) {
    std::ostringstream oss;
    oss << "Invalid Path insertion, "
        << "error detected in redundant variables. "
        << "Path destination is " << pathDest
        << ", which does not correspond to this TensorId (" << id << ").";
    throw error(oss.str());
  }

  for (const auto &alreadyPresent :
       inwardsPaths_[id.opId().get()][id.outIndex().get()]) {

    if (!alreadyPresent.dstRegions().disjoint(p.dstRegions())) {
      std::ostringstream oss;
      oss << "Inserting Path in unwind::Solution::"
          << "insertInwardsPath(id=" << id << ", Path=";
      p.append(oss);
      oss << ") but part of the Region is already covered. ";
      oss << "In particular, the Regions " << alreadyPresent.dstRegions()
          << " and " << p.dstRegions() << " intersect. ";
      throw error(oss.str());
    }
  }

  inwardsPaths_[id.opId().get()][id.outIndex().get()].push_back(p);
}

void Solution::insertPathBackToSink(const TensorId &tId, const Path &p) {
  pathsBackToSinks_[tId.opId().get()][tId.outIndex().get()].push_back(p);
}

const Paths &Solution::barriersToSinks() const { return barriersToSinks_; }

void Solution::append(std::ostream &ost) const {

  ost << "\nInwards Paths\n-----\n";
  for (auto tId : graph().tensorIds()) {
    for (const auto &p : inwardsPaths(tId)) {
      ost << p << '\n';
    }
  }
}

std::ostream &operator<<(std::ostream &ost, const Solution &soln) {
  soln.append(ost);
  return ost;
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
