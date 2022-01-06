// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_TESTUTIL_UNWIND_FULLSTATE_HPP
#define POPRITHMS_TESTUTIL_UNWIND_FULLSTATE_HPP

#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/memory/unwind/lower.hpp>
#include <poprithms/memory/unwind/scheduledsolution.hpp>

namespace poprithms {
namespace unwindtoy {

using poprithms::common::multiout::ConsumptionId;
using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using MultioutOp = poprithms::common::multiout::Op;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::memory::unwind::ScheduledSolution;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;
using poprithms::util::Permutation;
using Lower         = poprithms::ndarray::Shape::Lower;
using Upper         = poprithms::ndarray::Shape::Upper;
using HTensor       = poprithms::compute::host::Tensor;
using HTensors      = std::vector<HTensor>;
using SchedulableOp = poprithms::common::schedulable::Op;

using poprithms::memory::unwind::Path;

class Graph;

constexpr float unmappedValue{-146709.};

/**
 * Implements logic of Helper template class in getPathSrc.
 * */
class FullState {
public:
  FullState(const Graph &g);
  /**
   * The following ~10 methods are required to satisfy the API of
   * the Helper template class in unwind/lower.hpp
   *
   * 1) Return a the scheduled solution, consisting of Paths and Ops, to
   *    lower.
   * */
  const ScheduledSolution &scheduledSolution() const { return *ssp; }

  /**
   * 2) Lower the Op. This will call a backend function like poplin::matmul,
   *    to create backend code and tensors.
   * */
  void initialize(OpId opId);

  /**
   * 3, 4, 5) Methods to create unmapped sink tensors, and check if they
   *          exist.
   * */
  bool unwindSinkInitialized(const TensorId &tId) const;
  void initializeUnwindSink(const TensorId &pathDst);
  HTensor getUnwindSink(const TensorId &pathDst) const;

  /**
   * 6) Create an unmapped Tensor of Shape #s, using the Path p to determine
   *    additional attributes like numerical type. */
  HTensor createUnmapped(const Path &, const Shape &s) const;

  /**
   * 7) If the tensor in the unwind graph with id #uwId has already been
   *    initialized and has a complete layout, return it. Else return false.
   * */
  std::pair<bool, HTensor> finalLayout(const TensorId &uwId) const;

  /**
   * 8) Create an empty (unset) tensor. This tensor will never be used.
   * */
  HTensor createEmpty() const { return HTensor::int32(-1); }

  /**
   * 9) Create a mapped tensor for the source of the path #p. This method will
   * call into backend methods for creating specialized layouts, such as
   * poplibs' createLhsMatMul. #ins contains layouts which help determine the
   * layout of the source being mapped. An example of when #ins is used is
   * when the dominating broadcast input is used to map the dominated
   * broadcast input (for poplar's createBroadcastOperand).
   * */
  HTensor createMappedSrc(const Path &p, const HTensors &ins) const;

  /**
   * 10) Unwind the path #p from #src to #dst.
   * */
  void
  unwindAndUpdate(const Path &, const HTensor &src, const HTensor &dst) const;

  /**
   * The following methods are specific to this test class, and are not
   * required by the Helper API.
   * */
  TensorId toUnwind(const TensorId &id) const;
  TensorIds toUnwinds(const TensorIds &ids) const;
  TensorId toToy(const TensorId &tId) const;
  void insert(const TensorId &toy, const TensorId &uw);

  /**
   * For testing purposed: find an Op with all of the strings in 'frags' in
   * its name. If there is not a unique Op, throw an error.
   * */
  OpId unwindOpWithName(const std::vector<std::string> &frags) const;

  void lower();
  poprithms::memory::unwind::Graph &uwGraph() { return uwg_; }
  HTensor mainLayout(const TensorId &toyId) const;
  void setMainLayout(const TensorId &toyId, const HTensor &ht);
  HTensor createUnmapped(const Shape &s) const;
  HTensor createMappedSrc(const TensorId &uwId) const;

  // Testing utility
  HTensor createMappedSrc(const std::vector<std::string> &uwOpFrags,
                          OutIndex o) {
    return createMappedSrc({unwindOpWithName(uwOpFrags), o});
  }

private:
  poprithms::memory::unwind::Graph uwg_;
  std::map<TensorId, TensorId> toToy_;
  std::map<TensorId, TensorId> toUnwind_;
  std::map<TensorId, HTensor> mainLayouts_;
  std::map<TensorId, HTensor> unwindSinks_;
  const Graph &tg_;
  std::unique_ptr<ScheduledSolution> ssp;
  std::map<TensorId, HTensor> sources_;
};

} // namespace unwindtoy
} // namespace poprithms

#endif
