// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_IAUTOMATICMUTATOR_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_IAUTOMATICMUTATOR_HPP

#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/program/callstack/calleetensorid.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/carriedtensorid.hpp>
#include <poprithms/program/callstack/stackedio.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::ndarray::DType;
using poprithms::ndarray::Shape;
using poprithms::program::callstack::CalleeIndex;
using poprithms::program::callstack::CalleeTensorIds;
using poprithms::program::callstack::CarriedTensorIds;
using poprithms::program::callstack::IsStackedCopy;
using poprithms::program::callstack::StackedCopyOrder;

/**
 * Interface for a graph mutator capable of inserting certain operations
 * required for automatic differentiation into a graph. It is similar to a
 * poprithms::core::GraphMutator, but it has extensions for tensors with
 * shapes and types, and for operations which have callees.
 *
 * \sa automatic::Differentiator, the class which uses this interface.
 * */
class IAutomaticMutator {
public:
  IAutomaticMutator();
  virtual ~IAutomaticMutator();

  /**
   * Return a new view into the tensor #tId which has shape #s. The returned
   * tensor can be an alias of #tId (as suggested by the '_' suffix in its
   * name). It is allowed for values to be copied, however.
   * */
  virtual TensorId reshape_(const TensorId &tId, const Shape &s) = 0;

  /**
   * One-hot encode the tensor #t with a '1' at indices in #index and '0'
   * everywhere else. See Shape::assertOneHotEncodeable for shape
   * requirements. #index should be an integral tensor. The encoding must be
   * done inplace on #t, so that the returned tensor is an alias of #t.
   * */
  virtual TensorId encodeOneHot_(const TensorId &t,
                                 const TensorId &index) = 0;

  /**
   * Create a new, empty sub-graph with (optional) name #n.
   * */
  virtual SubGraphId createSubGraphId(const std::string &n) = 0;

  /**
   * Create a new variable which is like the tensor #tId in every respect
   * (sub-graph, device, etc.) other than type and shape. The new variable has
   * type #t and shape #s.
   * */
  virtual TensorId variableLike(const TensorId &tId,
                                DType t,
                                const Shape &s,
                                const std::string &n = {}) = 0;

  /**
   * Create a new variable which is like #tId in every respect other that the
   * sub-graph it belongs to. The returned variable is in sub-graph #sgId.
   * */
  virtual TensorId variableLike(const TensorId &tId,
                                SubGraphId sgId,
                                const std::string &n = {}) = 0;

  /**
   * Create a new rank-0 constant which is like tensor #tId (same type,
   * device, etc.) except for shape which is () and sub-graph which is #sgId.
   * */
  virtual TensorId scalarConstantLike(const TensorId &,
                                      SubGraphId sgId,
                                      double,
                                      const std::string &n = {}) = 0;

  /**
   * The sub-graph to which the op #opId belongs.
   * */
  virtual SubGraphId subGraphId(OpId opId) const = 0;

  /**
   * Perform a broadcasting view-change on #tId up to the shape #s. The shape
   * of #tId should be numpy-dominated by #s. See Shape::numpyDominates for
   * more information. This can be implemented by (1) unsqueeze_ followed
   * by individual calls to broadcast_ in each dimension.
   * */
  virtual TensorId expand_(const TensorId &, const Shape &) = 0;

  /**
   * Expand the singleton dimension #dim to be of size #N.
   * */
  virtual TensorId broadcast_(const TensorId &, uint64_t N, uint64_t dim) = 0;

  /**
   * Concatenate the tensors #tIds along dimension #dim. The returned tensor
   * is a view (alias) of the concatenated tensors.
   * */
  virtual TensorId concat_(const TensorIds &, uint64_t dim) = 0;

  /**
   * Set the values of #tId to zero, and return an alias of it.
   * */
  virtual TensorId zero_(const TensorId &tId) = 0;

  /**
   * Add the tensors #arg0 and #arg1 together, and return the result. The
   * add should not be inplace. This method should support numpy-broadcasting.
   * See https://numpy.org/doc/stable/user/basics.broadcasting.html for
   * broadcasting rules.
   * */
  virtual TensorId add(const TensorId &, const TensorId &) = 0;

  /**
   * Remove the op #opId from the graph. Consumers of #opId's outputs should
   * consume #replacements instead after #opId is removed. Add (optional)
   * context via #s for improved debugging.
   * */
  virtual void removeOp(OpId opId,
                        const OptionalTensorIds &replacements,
                        const std::string &context) = 0;

  /**
   * Insert a call operation into the sub-graph caller, which calls the
   * sub-graph #callee. Each pair in #ins contains (1) a tensor in the caller
   * and (2) a tensor in the callee (of same type and shape). These copies
   * from (1) to (2) are performed before #callee is executed. The outputs in
   * #outs are the tensors in the callee to be copied out.
   * */
  virtual OpId call(SubGraphId caller,
                    SubGraphId callee,
                    const std::vector<std::pair<TensorId, TensorId>> &ins,
                    const TensorIds &outs) = 0;

  /**
   * Insert a repeat operation in the sub-graph #caller, which repeats the
   * sub-graph #callee for #rptCount iterations.
   *
   * The inputs are #stackedIns and #carriedIns:
   *
   * Every pair in #stackedIns is made up of (1) a tensor in #caller and (2) a
   * tensor in #callee. The tensor in #caller should have the shape of the
   * tensor in #callee but with #rptCount prepended. For example if the tensor
   * in #callee has shape (3) and #rptCount is 5, the tensor in #caller
   * has shape (5,3). At every iteration of the repeat loop, one slice from
   * the tensor in #caller will be copied to the tensor in #callee. The slices
   * either proceed in ascending order (if copyOrder is Up) or in descending
   * order (if copyOrder is Down).
   *
   * The inputs in #carriedIns are the tensors which are loop carry
   * dependencies. See the CarriedTensorId for more information.
   *
   * The outputs in #outs are tensors in the callee graph. For them, either
   * just the final value after the final iteration is returned (when
   * IsStackedCopy is No) or the value at every iteration is returned (when
   * IsStackedCopy is Yes). The order of the outputs when IsStackedCopy is
   * true is dictated by #copyOrder.
   * */
  virtual OpId
  repeat(SubGraphId caller,
         SubGraphId callee,
         uint64_t rptCount,
         const std::vector<std::pair<TensorId, TensorId>> &stackedIns,
         const CarriedTensorIds &carriedIns,
         const std::vector<std::pair<TensorId, IsStackedCopy>> &outs,
         StackedCopyOrder copyOrder) = 0;

  /**
   * Insert a switch operation into the sub-graph #caller, which conditionally
   * (conditional on the value of the scalar tensor #condition) executes one
   * of the sub-graphs in #callees.
   *
   * The inputs are defined by #ins. Elements of #ins are make up of 3
   * components: (1) a tensor in the caller graph (the source of the copy) and
   * (2) a tensor in the callee graph at index (3) of #callees.
   *
   * The outputs are separated into 2 groups, the first is #completeOuts. The
   * tensor completeOuts[outIndex][calleeIndex] is the tensor which is copied
   * out of the callee sub-graph #calleeIndex at output index #outIndex if
   * #condition is #calleeIndex. That is, this out-copy is performed
   * conditionally on #condition. Specifically, exactly 1 copy is performed
   * for each #outIndex.
   *
   * The second is #unmergedOuts. #unmergedOuts[outIndex] is a group of
   * tensors (of size less than or equal to the number of callees). Unlike
   * #completeOuts, it is possible for #unmergedOuts[outIndex] to not have any
   * tensors for some callee sub-graphs, in which case no copy out is
   * performed for these indices.
   * */
  virtual OpId switchOp(
      SubGraphId caller,
      const SubGraphIds &callees,
      const TensorId &condition,
      const std::vector<std::tuple<TensorId, TensorId, CalleeIndex>> &ins,
      const std::vector<std::vector<TensorId>> &completeOuts,
      const std::vector<CalleeTensorIds> &unmergedOuts) = 0;

  /**
   * The shape of tensor #tId.
   **/
  virtual Shape shape(const TensorId &tId) const = 0;

  TensorId unsqueeze_(const TensorId &tId, uint64_t d) {
    return reshape_(tId, shape(tId).unsqueeze(d));
  }

  TensorId scalarConstantLike(const TensorId &tId,
                              double v,
                              const std::string &n = {}) {
    return scalarConstantLike(tId, subGraphId(tId.opId()), v, n);
  }

  /**
   * Create a broadcast scalar.
   * */
  TensorId
  zeroLike(const TensorId &tId, SubGraphId sgId, const std::string &n = {});
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
