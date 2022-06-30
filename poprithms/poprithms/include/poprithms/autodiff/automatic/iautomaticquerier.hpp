// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_IAUTOMATICQUERIER_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_IAUTOMATICQUERIER_HPP

#include <array>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/autodiff/core/gradinfo.hpp>
#include <poprithms/autodiff/core/graphmutator.hpp>
#include <poprithms/autodiff/core/summary.hpp>
#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/autodiff/guide/guide.hpp>
#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/optraversal.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/program/callstack/calleetensorid.hpp>
#include <poprithms/program/callstack/carriedtensorid.hpp>
#include <poprithms/program/callstack/copyout.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

using poprithms::autodiff::guide::Objective;
using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::ndarray::Shape;
using poprithms::program::callstack::CalleeTensorId;
using poprithms::program::callstack::CalleeTensorIds;
using poprithms::program::callstack::CallEvent;
using poprithms::program::callstack::CallEvents;

/**
 * Interface for a graph querier. It makes queries which are specific to
 * automatic differentiation. \sa automatic::Differentiator, the class which
 * uses this interface.
 * */
class IAutomaticQuerier {
public:
  IAutomaticQuerier()          = default;
  virtual ~IAutomaticQuerier() = default;

  /**
   * The number of input tensors of the op #opId.
   * */
  virtual uint64_t nInTensors(OpId opId) const = 0;

  /**
   * The number of output tensors of the op #opId.
   * */
  virtual uint64_t nOutTensors(OpId opId) const = 0;

  /**
   * The number of inputs which are copies into sub-graphs.
   * */
  virtual uint64_t nInCopies(OpId) const = 0;

  /**
   * \return The callee which the input at index #i of the op #opId is copied
   * to.
   * */
  virtual CalleeIndex inDstCalleeIndex(OpId, InIndex) const = 0;

  /**
   * For the calling op #opId, what is the destination in a callee sub-graph
   * of the input at #inIndex ?
   * */
  virtual CalleeTensorId inDst(OpId opId, InIndex inIndex) const = 0;

  /**
   * The source of the copy out of the callee #ci of op #opId, at output index
   * #outIndex.
   * */
  virtual TensorId
  outSource(OpId opId, OutIndex outIndex, CalleeIndex ci) const = 0;

  /**
   * The sources of the copies out of the callee #ci of op #opId, at output
   * indices #outIndices.
   * */
  TensorIds
  outSources(OpId opId, CalleeIndex ci, const OutIndices &outIndices) const;

  /**
   * \return true if the tensor #tId is copied out of the sub-graph #ci of the
   *         op #opId.
   * */
  virtual bool
  isOutSource(OpId opId, CalleeIndex ci, const TensorId &tId) const = 0;

  /**
   * \return The output index at which the callee sub-graph tensor, in the
   *         callee sub-graph #ci of the calling op #opId, is copied out at.
   * */
  virtual OutIndex
  copyOutIndex(OpId opId, CalleeIndex ci, const TensorId &tId) const = 0;

  /**
   * The tensor #tId is a tensor in one of the callee sub-graphs of the call
   * event #ce. Where is it copied to in the caller sub-graph?
   * */
  virtual TensorId dstInCaller(const TensorId &tId,
                               const CallEvent &ce) const = 0;

  /**
   * The sub-graph of op #opId.
   * */
  virtual SubGraphId subGraphId(OpId opId) const = 0;

  /**
   * \return true if the tensor #tId consist only of constant 0. Backends are
   *         free to just return false for this. It is used only in an edge
   *         case of differentiating a repeat op.
   * */
  virtual bool isDefinitelyAllConstZero(const TensorId &tId) const = 0;

  /**
   * All tensors in the sub-graph #sgId.
   * */
  virtual TensorIds tensorIds(SubGraphId sgId) const = 0;

  /**
   * The number of callees of the op #opId.
   * */
  virtual uint64_t nCallees(OpId opId) const = 0;

  /**
   * The #ci'th callee of the op #opId.
   * */
  virtual SubGraphId callee(OpId, CalleeIndex) const = 0;

  /**
   * The shape of the tensor #tId.
   * */
  virtual Shape shape(const TensorId &tId) const = 0;

  /**
   * A string representation of op #opId. This is used for logging and
   * improved error messages.
   * */
  virtual std::string str(OpId) const = 0;

  /**
   * Create an objective of differentiation (see the class guide::Objective)
   * for differentiating the #callIndex'th sub-graph of the op #opId.
   *
   * The objective must create gradients for the inputs of #opId at
   * #fromTargets, and must take in gradients for the outputs at indices
   * #gradsIn.
   * */
  virtual Objective localObjective(OpId opId,
                                   CalleeIndex callIndex,
                                   const InIndices &fromTargets,
                                   const OutIndices &gradsIn) const = 0;

  /**
   * \sa guide::GraphInfo::gradientPropagates.
   * */
  virtual bool gradientPropagates(OpId, OutIndex, InIndex) const = 0;

  /**
   * The input of op #opId at input index #inIndex.
   * */
  virtual TensorId inTensorId(OpId opId, InIndex inIndex) const = 0;

  /**
   * All consumers of the tensor #tId.
   * */
  virtual ConsumptionIds consumptionIds(const TensorId &tId) const = 0;

  /// The following non-virtual methods are implemented in terms of the
  /// virtual methods above.

  TensorId inTensorId(const OpTraversal &ot) const {
    return inTensorId(ot.opId(), ot.inIndex());
  }

  bool gradientPropagates(const OpTraversal &ot) const {
    return gradientPropagates(ot.opId(), ot.outIndex(), ot.inIndex());
  }

  TensorId outTensorId(const OpTraversal &ot) const {
    return {ot.opId(), ot.outIndex()};
  }

  TensorIds inTensorIds(OpId) const;

  TensorIds outTensorIds(OpId) const;

  TensorId outTensorId(OpId opId, OutIndex o) const { return {opId, o}; }

  uint64_t nelms_u64(const TensorId &tId) const {
    return shape(tId).nelms_u64();
  }

  SubGraphId subGraphIdFromTensorIds(const std::vector<TensorIds> &) const;

  SubGraphId subGraphIdFromObjective(const Objective &o) const;

  TensorIds inDsts(OpId, const InIndices &) const;

  virtual SubGraphId subGraphId(const TensorId &tId) const {
    return subGraphId(tId.opId());
  }

  /**
   * The callees of the op #opId.
   * */
  SubGraphIds callees(OpId opId) const;

  virtual CallEvent event(OpId opId, CalleeIndex ci) const {
    return CallEvent(opId, callee(opId, ci), ci);
  }

  SubGraphId subGraphIdFromTensorIds(const TensorIds &tIds) const;
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
