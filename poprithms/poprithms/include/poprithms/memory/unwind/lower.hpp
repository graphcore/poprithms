// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_LOWER_HPP
#define POPRITHMS_MEMORY_UNWIND_LOWER_HPP

#include <poprithms/memory/unwind/scheduledsolution.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

/**
 * Template classes are
 *
 * T: a 'device' tensor. For the main application of the project, this will be
 *    a poplar::Tensor, but it is templatized to make unit testing without
 *    poplar possible.
 *
 * Helper: a class for manipulating the 'T' template class. The methods of
 *         Helper are docmumented in the testutil class, FullState.
 *
 * */

template <typename T, typename Helper> class Lowerer {

public:
  /** Lower the ops and unwind paths defined in #h to the backend. For
   * example, this might create a poplar Graph and give all tensors a complete
   * tile mapping. To see what methods the Helper class needs to implement,
   * see the testutil class, FullState. */
  static void lower(Helper &h) {
    Lowerer<T, Helper> x(h);
    x.lower();
  };

  void lower() {
    const ScheduledSolution &ss = helper.scheduledSolution();

    /**
     * ScheduledSolution ss is a sequence of 'nodes' corresponding to either
     * (1) an op in the compute graph or (2) a path (a chain of view-changes)
     * between tensors. The nodes are ordered in topological order, so that
     * they can be lowered in order without missing any dependencies.
     * */
    for (auto n : ss.schedule()) {

      // If the node is an op, call 'initialize' on it (similar to popart's
      // grow method).
      if (ss.isOp(n)) {
        helper.initialize(ss.op(n));
      }
      // If the node is a path, then unwind from the source to the
      // destination. This gives the destination of the path a layout/mapping.
      else {
        const auto &p   = ss.pathToSink(n);
        const auto tSrc = getPathSrc(p);
        if (!helper.unwindSinkInitialized(p.dst())) {
          helper.initializeUnwindSink(p.dst());
        }
        const auto tDst = helper.getUnwindSink(p.dst());
        helper.unwindAndUpdate(p, tSrc, tDst);
      }
    }
  }

private:
  Helper &helper;
  Lowerer(Helper &h) : helper(h) {}

  std::pair<bool, T> layout(const TensorId &uwId) {
    /**
     * Look in 2 places for a T. First, check if the #uwId corresponds to a
     * final tensor in the 'compute' graph with a known layout:
     * */
    auto r = helper.finalLayout(uwId);
    if (r.first) {
      return r;
    }

    /**
     * Second, check if there is a cached T for #uwId. If there is not, then
     * {false, empty-T} is returned.
     * */
    return cachedLayout(uwId);
  }

  void insertCacheSrc(const Path &p, const T &t) {
    cache_.insert({p.src(), t});
  }

  std::pair<bool, T> cachedLayout(const TensorId &uwId) const {

    auto f1 = cache_.find(uwId);
    if (f1 != cache_.cend()) {
      return {true, f1->second};
    }

    return {false, helper.createEmpty()};
  }

  /**
   * Get the tensor, with complete layout, at the start of the Path #p. Use
   * #helper to translate between the "compute" graph and the "unwind" graph,
   * and to manage caching.
   * */
  T getPathSrc(const poprithms::memory::unwind::Path &p) {

    // The op at the start of the path.
    const auto barrierOp = p.src().opId();

    // If there is a T with a known 'layout' corresponding to the source of
    // the path, then return it. Having this initial check means that caching
    // can reduce the total amount of backend tensor creation required.
    const auto easyFind0 = layout(p.src());
    if (easyFind0.first) {
      return easyFind0.second;
    }

    const auto &ss  = helper.scheduledSolution();
    const auto &uwg = ss.graph();

    // A function to get the T for an unwind tensor, uwIn.
    const auto getIn = [&uwg, &ss, this, &p](const TensorId &uwIn) {
      const auto easyFind1 = layout(uwIn);
      if (easyFind1.first) {
        return easyFind1.second;
      }

      else {
        const Shape shape = uwg.shape(uwIn);
        T inProxy         = helper.createUnmapped(p, shape);
        for (auto p2 : ss.inwardsPaths(uwIn)) {
          // note: recursive function call.
          T subBarrier = getPathSrc(p2);
          helper.unwindAndUpdate(p2, subBarrier, inProxy);
        }
        return inProxy;
      }
    };

    // We couldn't find a cached T for the output of barrierOp (p.src()), so
    // we need to create a T. As barrierOp is a barrier op, it has an
    // associated function to create a T -- we will create this with
    // helper.createMappedSrc. However, there might be some additional T's
    // required to create a T for barrierOp. For example, if barrierOp creates
    // a (dominated) T for a broadcast add, it will require the (dominating) T
    // which the operand gets added.
    //
    // We start by collecting these input Ts to the barrier op:
    std::vector<T> srcIns;
    srcIns.reserve(uwg.nInTensors(barrierOp));
    for (auto uwIn : uwg.inTensorIds(barrierOp)) {
      srcIns.push_back(getIn(uwIn));
    }

    auto out = helper.createMappedSrc(p, srcIns);
    insertCacheSrc(p, out);
    return out;
  }

private:
  std::map<TensorId, T> cache_;
};

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
