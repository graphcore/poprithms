// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_GRAPH_HPP
#define POPRITHMS_COMMON_MULTIOUT_GRAPH_HPP

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/fwdedgemap.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/optraversal.hpp>
#include <poprithms/common/multiout/removalevent.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/copybyclone.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace multiout {

using ndarray::Shape;
using ndarray::Shapes;

/**
 * A class to represent a Graph where the nodes (Ops) can have multiple Tensor
 * inputs and Tensor outputs.
 *
 * The Ops input Tensors are at contiguous input indices (InIndices), and the
 * output Tensors are at contiguous output indices (OutIndex).
 *
 * There are no explicit control dependencies, only implicit data control
 * dependencies implied by Tensors.
 * */
class Graph {

public:
  Graph()          = default;
  virtual ~Graph() = default;

  /**
   * Graph constructors and assignment operators. The reason these are not
   * 'default' is that Ops contain (const) pointers to their containing
   * Graphs. When a Graph #a is copied to Graph #b, all of the Ops of #a are
   * cloned into #b, and then the clones have their pointers updated to #b.
   * */
  Graph(Graph &&);
  Graph(const Graph &);
  Graph &operator=(Graph &&);
  Graph &operator=(const Graph &);

  /** Set the name of the Op #id in this Graph. */
  void setName(OpId id, const std::string &);
  void setName(const TensorId &id, const std::string &);

  /** The name of an Op #id in this Graph. */
  const std::string &getName(OpId id) const;

  /** The Shape of a Tensor #id in this Graph. */
  Shape shape(const TensorId &id) const;

  /** Shapes of multiple Tensors in this Graph. */
  Shapes shapes(const TensorIds &ids) const;

  /** The number of elements of a Tensor #x in this Graph. */
  uint64_t nelms_u64(const TensorId &x) const { return shape(x).nelms_u64(); }

  int64_t nelms(const TensorId &x) const { return shape(x).nelms(); }

  /** The rank of a Tensor in this Graph. */
  uint64_t rank_u64(const TensorId &x) const { return shape(x).rank_u64(); }

  /** All Consumers of a Tensor #id in this Graph. */
  ConsumptionIds consumptionIds(const TensorId &id) const;

  /** DAG of all the (data) edges in this Graph. */
  FwdEdgeMap getMultioutForwardEdgeMap_u64() const;

  /**
   * DAG of a subset of the (data) edges in this Graph. This is equivalent
   * to, but more efficient than, finding the complete edge map for all ops,
   * and removing all entries whose keys are not in a data-connected component
   * of an op in #mustInclude. Or in other words, it is the DAG of all ops in
   * the connected components of #mustInclude.
   * */
  FwdEdgeMap getMultioutForwardEdgeMap_u64(const OpIds &mustInclude) const;

  /**
   * The number of ConsumptionIds that the Tensor #id has. This is not
   * necessarily the number of Ops which consume Tensor #id, but is an upper
   * bound of that number. This is because Ops can consume the same Tensor at
   * multiple InIndexes. As an example, suppose that Tensor #id is the input
   * to Op #op0 at InIndex 0, and the input to Op #op1 at InIndexes 0 and 1.
   * This method will return 3, as there are 3 consumption 'sites', even
   * though only 2 Ops consume #id.
   * */
  uint64_t nConsumptionIds(const TensorId &id) const;

  /**
   * Return true if the Tensor #id is the input to any Op.
   * */
  bool hasConsumptionIds(const TensorId &id) const {
    return nConsumptionIds(id) != 0;
  }

  /** Set the name of this Graph. */
  void setName(const std::string &n) { atts.name_ = n; }

  /** The total number of Tensors in this Graph. */
  uint64_t nTensors() const { return nOutTensors(opIds()); }
  uint64_t nOutTensors(const OpIds &) const;

  /** The total number of Ops in this Graph. */
  uint64_t nOps() const { return atts.live_.size(); }

  /**
   * If an Op #opId was created and not yet removed, return true. Otherwise,
   * return false.
   * */
  bool isLive(OpId opId) const { return atts.live_.count(opId) != 0; }

  /** The total number of Ops in this Graph which have 0 outputs. */
  uint64_t nOpsWithZeroOutputs() const;
  uint64_t nWithZeroOutputs(const OpIds &) const;

  int64_t nOps_i64() const { return static_cast<int64_t>(atts.live_.size()); }

  /** \return The number of inputs of the Op #id.*/
  uint64_t nInTensors(OpId id) const;

  /** \return The number of outputs of the Op #id.*/
  uint64_t nOutTensors(OpId id) const;

  /** \return The Shapes of the inputs of the Op #id. */
  Shapes inShapes(OpId id) const;

  /** \return The Shapes of the outputs of the Op #id. */
  Shapes outShapes(OpId id) const;

  /** \return All InIndices of Op #id. These are [0, ..., nInTensors(opId)).
   */
  InIndices inIndices(OpId id) const;

  /** \return All OutIndices of Op #id. These are [0, ..., nOutTensors(opId)).
   */
  OutIndices outIndices(OpId) const;

  /**
   * The output TensorIds of Op #id.
   * These are simply TensorId(id, o) for o in [0, nOutTensors).
   * */
  TensorIds outTensorIds(OpId id) const;
  TensorId outTensorId(OpId id, OutIndex o) const { return {id, o}; }

  /**
   * The TensorIds of the inputs of Op #id.
   * */
  TensorIds inTensorIds(OpId id) const;
  TensorId inTensorId(OpId, InIndex) const;

  /**
   * The vector-concatenation of the TensorIds of all input and output
   * Tensors. For example. if the input TensorIds of the Op with opId=3 are
   * ((opId=0,outIndex=0), (0,0), (2,1)), and the outputs are ((3,0), (3,1)),
   * then the returned vector is simply ((0,0), (0,0), (2,1), (3,0), (3,1)).
   * */
  TensorIds inAndOutTensorIds(OpId) const;

  /** \return The string description of the Op #id. */
  std::string typeString(OpId id) const;

  /**
   * Verify that there is a Tensor with TensorId #tId in this Graph. If there
   * is not, a descriptive error is thrown.
   * */
  void verifyTensorId(const TensorId &tId) const;

  /** The name of this Graph. */
  const std::string &getName() const { return atts.name_; }

  /**
   * We implement operator== once in this base class, and use the non-virtual
   * interface (NVI) idiom for derived classes to specify equivalence: the
   * pure virtual function, #multiOutTypeSpecificEqualTo, must be implemented
   * by derived classes to perform the equality check, so that this base class
   * doesn't need to know what it means for derived classes to be equivalent.
   * */
  bool operator==(const Graph &rhs) const;
  bool operator!=(const Graph &rhs) const { return !operator==(rhs); }

  /** All Op names, pythonically: [op(i).name() for i in range(nOps())]/ */
  std::vector<std::string> getOpNames() const;

  /**
   * Strip the OpIds from TensorIds. These are the Ops which create the
   * Tensors
   * */
  static OpIds opIds(const TensorIds &tids);

  /** In set notation: a \ b */
  static TensorIds setDifference(const TensorIds &a, const TensorIds &b);

  /** The TensorIds of all (live) Tensors in this Graph */
  TensorIds tensorIds() const;

  /** The OpIds of all (live) Ops in this Graph */
  OpIds opIds() const {
    return OpIds(atts.live_.cbegin(), atts.live_.cend());
  }

  /**
   * Consider a table summarising the Graph, where for each Tensor, and for
   * each Op without any output, there is one row of information.
   *
   * This method returns the columns of such a table. The table might be,
   *
   *  OpId OpType      InTensors OutIndex Shape
   *  ---- ------      --------- -------- -----
   *  0    Foo         ()
   *  1    Bar         ()        0        (1)
   *                             1        (1,2)
   *                             2        (1,1,1)
   *
   * in the case where there are 2 Ops, one with 0 outputs and one with 3
   * outputs. Each of the 5 columns will then have 6 strings in it. The first
   * column will have
   * ("OpId ", "---- ", "0    ", "1    ", "     ", "     ", "     "), etc.
   *
   * The spacing, column width, and other aspects of the layout of the
   * columns are controlled by #format.
   *
   * */
  std::vector<poprithms::util::StringColumn> getMultioutColumns(
      const poprithms::util::StringColumn::Parameters &format) const;

  /**
   * Get the multiout columns (see above) of a subset of the Ops.
   * */
  std::vector<poprithms::util::StringColumn>
  getMultioutColumns(const OpIds &,
                     const poprithms::util::StringColumn::Parameters &) const;

  /**
   * Append a summary of the Ops in #opIds to the stream #ost.
   * */
  virtual void appendOpColumns(std::ostream &ost, const OpIds &) const = 0;

  void append(std::ostream &ost) const { appendOpColumns(ost, opIds()); }

  /**
   * \sa getMultioutColumns
   *
   * There is an entry (a row) for
   *  1) every Tensor,
   *  2) every Op which has no outputs.
   * */
  uint64_t nMultioutRows() const {
    return nTensors() + nOpsWithZeroOutputs();
  }

  /**
   * The number of columns (see above) for a subset of all Ops.
   * */
  uint64_t nMultioutRows(const OpIds &) const;

  /**
   * The OpTraversal #opTraversal consists of
   * (1) an input index #i
   * (2) an op #op, and
   * (3) an output index,
   * This method returns the tensor which is the input #i to the op #op.
   * */
  TensorId inTensorId(const OpTraversal &) const;

  static TensorId outTensorId(const OpTraversal &o);

  /**
   * Confirm that the Tensor #tId is in this Graph. Specifically, that the Op
   * which creates #tId exists and has an #tId as an output. If not, a
   * descriptive error is thrown.
   * */
  void verifyValidTensorId(const TensorId &tId) const;

  /**
   * \return A string summarizing the Ops which have been removed.
   *
   * When Ops are removed, a record of the removal 'event' is kept, with
   * optional information about the transformation which removed it. This
   * method provides a summary of all such removal events, and is used for
   * improved logging and debugging.
   * */
  std::string removalEventsStr() const { return atts.removals_.str(); }

  /**
   * \return The first (lowest) OpId which has not been used for an Op.
   **/
  OpId nxtOpId() const { return atts.ops_.size(); }

  /**
   * The output indices of all Tensors created by #opId, which are consumed by
   * an Op.
   * */
  std::vector<OutIndex> outIndicesConsumed(OpId opId) const;

  /** The sequence of Op removal events. */
  const RemovalEvents &removalEvents() const { return atts.removals_; }

  /**
   * Ops in this Graph contain a pointer to Graph. They should all point to
   * this Graph. Verify that this is the case.
   * */
  void verifyOpsConnectedToThisGraph() const;

protected:
  /**
   * Insert \a op into this Graph, and add it to the consumer lists of its
   * inputs' creators.
   * */
  OpId insertMultioutOp(std::unique_ptr<Op> op);

  [[noreturn]] void unimplemented(const std::string & = {}) const;

public:
  /**
   * Remove the Op #opToRemove from this Graph.
   *
   * The consumers of #opToRemove's output tensors need substitutes for their
   * inputs, which will no longer exist with the removal of #opToRemove. These
   * substitutes are provided in #outputSubstitutes. outputSubstitutes[i] may
   * only be 'none' if nConsumptionIds(opToRemove, i) is 0. That is, if the
   * i'th output of #opToRemove has any consumers, then a replacement must be
   * provided.
   *
   * The optional string #removalContext is used for logging and debugging
   * purposes. After #opToRemove has been removed from the set of 'live' ops,
   * a lightweight record of it and its removal events is retained. This
   * record is particularly useful if there is an attempted access of
   * #opToRemove after it has been removed. Such an attempt will result in a
   * descriptive error, including the #removalContext string.
   *
   * This method calls into a virtual method of the Op class to perform the
   * changes required for derived classes.
   * */
  void removeOp(OpId opToRemove,
                const OptionalTensorIds &outputSubstitutes,
                const std::string &removalContext);

  /**
   * Remove the inputs at indices #toRemove from the Op #toPrune. The
   * remaining inputs are shifted to lower indices so as to occupy vacated
   * input indices, resulting in contiguous inputs from index 0.
   *
   * This method calls into #multiOutTypeSpecificRemoveInputs to perform the
   * required changes for derived Graph classes.
   * */
  void removeInputs(OpId toPrune, const InIndices &toRemove);

private:
  /**
   * \sa removeInputs.
   *
   * Perform the removal work of derived graph classes, this method is called
   * into by #removeInputs.
   *
   * \param toPrune the op whose inputs must be removed.
   *
   * \param coin describes which input indices should have their inputs
   *             removed. This object provides utility methods for removing
   *             elements from vectors are the specified indices.
   * */
  virtual void
  multiOutTypeSpecificRemoveInputs(OpId toPrune,
                                   const ContiguousInIndexSubset &coin) = 0;

public:
  void replaceInput(OpId toChange, InIndex, const TensorId &sub);

  /**
   * Remove the outputs at indices #toRemove from the Op #toPrune.
   *
   * \param outputSubstitutes the Tensors to replace the removed output
   *                          Tensors. Specifically, ops which consume the
   *                          outputs of #toPrune at indices #toRemove will
   *                          consume #outputSubstitutes instead. See
   *                          #removeOp for more information on how
   *                          substitutes work.
   *
   * The outputs at retained output indices are shifted down to fill the gaps
   * created. For example, if output 0 is removed, then output 1 becomes the
   * new output at index 0. This change in indices is propagated to all
   * consumers.
   *
   * This method calls #multiOutTypeSpecificRemoveOutputs to perform the work
   * of inheriting graph classes.
   * */
  void removeOutputs(OpId toPrune,
                     const OutIndices &toRemove,
                     const OptionalTensorIds &outputSubstitutes);

private:
  /**
   * \sa removeOutputs, multiOutTypeSpecificRemoveInputs.
   * */
  virtual void
  multiOutTypeSpecificRemoveOutputs(OpId,
                                    const ContiguousOutIndexSubset &,
                                    const OptionalTensorIds &subs) = 0;

private:
  /**
   * The creator of the input #index of #op stores the fact that #op consumes
   * its output. This method removes this consumption id. #op is left
   * unchanged.
   * */
  void removeConsumptionId(OpId op, InIndex index);

  /**
   * This method is used when an input of #op moves from #oldIndex to
   * #newIndex. The creator of the input at old index has its consumption ids
   * updated from (op, oldIndex) to (op, newIndex). #op is left unchanged.
   */
  void resetConsumption(OpId op, InIndex oldIndex, InIndex newIndex);

public:
  // Classes which inherit from multiout::Graph might have some additional
  // steps when removing an op. These are performed in this method.
  virtual void multiOutTypeSpecificRemoveOp(
      OpId opToRemove,
      const OptionalTensorIds &outputSubstitutes) = 0;

  /**
   * Verify that 'after' is a valid replacement for 'before'. For example,
   * 'before' and 'after' must have the same Shape. Derived classes might have
   * Tensors with additional attributes, for which
   * 'multiOutTypeSpecificVerifyValidSubstitute' defines whether a replacement
   * is valid.
   * */
  void verifyValidSubstitute(const TensorId &before,
                             const TensorId &after) const;

  void verifyValidSubstitutesForRemoval(
      OpId toRemove,
      const OptionalTensorIds &outputSubstitutes) const;

  /** \sa verifyValidSubstitute */
  virtual void
  multiOutTypeSpecificVerifyValidSubstitute(const TensorId &before,
                                            const TensorId &after) const = 0;

  Op &multioutOp(OpId id) { return op(id); }
  const Op &multioutOp(OpId id) const { return op(id); }

  /**
   * Verify that all aspects of the graph are correct. For this (base class)
   * graph there are checks that the consumer-producer relationships are all
   * valid. Derived graph classes implement checks in
   * #verifyMultioutDerivedGraphValid, which this method calls into.
   * */
  void verifyValid() const;

  /**
   * Verify that the op #opId is valid, at every level of inheritance. This
   * method uses the same inheritance design as #verifyValid, but for just
   * a single op instead of the entire graph.
   * */
  void verifyOpValid(OpId) const;

  /**
   * \return All tensors which are on a data path to one or several of the
   *         tensors in #ids. The returned set includes #ids.
   * */
  TensorIds onPathTo(const TensorIds &ids) const;

private:
  /**
   * Derived classes must define what it means to be equivalent in this
   * virtual method.
   * */
  virtual bool multiOutTypeSpecificEqualTo(const Graph &) const = 0;

  /**
   * Verify that the attributes of this multiout op are correct.
   * */
  void verifyValidAtMultioutLevel(OpId) const;

  /**
   * Verify that all derived op attributes are correct.
   * */
  virtual void verifyMultioutDerivedOpValid(OpId opId) const = 0;

  /**
   * \sa verifyValid.
   * */
  virtual void verifyMultioutDerivedGraphValid() const = 0;

  /** Methods to access Ops in this Graph as multiout::Ops. */
  Op &nonConstMultioutOp(OpId id) { return op(id); }

  /**
   * Return the  id'th Op in the member variable \a ops, performing checks
   * that 0 <= id < nOps().
   * */
  Op &op(OpId id);
  const Op &op(OpId) const;

  /** All of the class data is stored in this default copyable class. */
  class Attributes {

  public:
    bool operator==(const Attributes &) const;

    /** All of the Ops in this Graph. The Ops are stored as unique_ptrs
     * wrapped in the CopyByClone class, which makes them, and thus the class,
     * copyable. When this Graph is copied, the resulting copy has a clone of
     * all of the Ops in this Graph.
     */
    std::vector<poprithms::util::CopyByClone<Op>> ops_;

    /**
     * The Ops which have not been deleted.
     * */
    std::set<OpId> live_;

    /**
     * Every OpId in [0, ops_.size()) corresponds to either a 'live' Op, or to
     * an op which once was live, but has been removed. If it was removed, a
     * record of it and it's removal is kept. This object stores these
     * records.
     * */
    RemovalEvents removals_;

    // The name of this Graph.
    std::string name_;
  } atts;

  const std::vector<poprithms::util::CopyByClone<Op>> &ops() const {
    return atts.ops_;
  }

  std::vector<poprithms::util::CopyByClone<Op>> &ops() { return atts.ops_; }

  /** Set the Graph pointed to by all Ops in this Graph, to this Graph. */
  void resetGraphOfOps();
};

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
