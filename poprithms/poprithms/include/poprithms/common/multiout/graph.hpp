// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_GRAPH_HPP
#define POPRITHMS_COMMON_MULTIOUT_GRAPH_HPP

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <poprithms/common/multiout/consumptionid.hpp>
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
  Graph()              = default;
  virtual ~Graph()     = default;
  Graph(Graph &&)      = default;
  Graph(const Graph &) = default;
  Graph &operator=(Graph &&) = default;
  Graph &operator=(const Graph &) = default;

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

  /** Set the name of this Graph. */
  void setName(const std::string &n) { name_ = n; }

  /** The total number of Tensors in this Graph. */
  uint64_t nTensors() const { return nOutTensors(opIds()); }
  uint64_t nOutTensors(const OpIds &) const;

  /** The total number of Ops in this Graph. */
  uint64_t nOps() const { return live_.size(); }

  /** The total number of Ops in this Graph which have 0 outputs. */
  uint64_t nOpsWithZeroOutputs() const;
  uint64_t nWithZeroOutputs(const OpIds &) const;

  int64_t nOps_i64() const { return static_cast<int64_t>(live_.size()); }

  /** \return The number of inputs of the Op #id.*/
  uint64_t nInTensors(OpId id) const;

  /** \return The number of outputs of the Op #id.*/
  uint64_t nOutTensors(OpId id) const;

  /** \return The Shapes of the inputs of the Op #id. */
  Shapes inShapes(OpId id) const;

  /** \return The Shapes of the outputs of the Op #id. */
  Shapes outShapes(OpId id) const;

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
   * The concatenation of the TensorIds of all input and output Tensors.
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
  const std::string &getName() const { return name_; }

  /**
   * We implement operator== once in this base class, and use the non-virtual
   * interface (NVI) idiom for derived classes to specify equivalence: the
   * pure virtual function, #typeSpecificEqualTo, must be implemented by
   * derived classes to perform the equality check, so that this base
   * class doesn't need to know what it means for derived classes to be
   * equivalent.
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

  /** The TensorIds of all Tensors in this Graph */
  TensorIds tensorIds() const;

  /** The OpIds of all Ops in this Graph */
  OpIds opIds() const;

  /**
   * Consider a table summarising the Graph, where for each Tensor, and for
   * each Op without any output, there is a row of information.
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
   * if there are 2 Ops, one with 0 outputs and one with 3 outputs. Each of
   * the 5 columns will then have 6 strings in it. The first column will have
   * ("OpId ", "---- ", "0    ", "1    ", "     ", "     ", "     "), etc.
   *
   * */
  std::vector<poprithms::util::StringColumn> getMultioutColumns() const;

  /**
   * Get the multiout columns (see above) of a subset of the Ops.
   * */
  std::vector<poprithms::util::StringColumn>
  getMultioutColumns(const OpIds &) const;

  /**
   * Append a summary of the the Ops in #opIds to the stream #ost.
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
  void confirmValidTensorId(const TensorId &tId) const;

  /**
   * \return A string summarizing the Ops which have been removed.
   *
   * When Ops are removed, a record of the removal 'event' is kept, with
   * optional information about the transformation which removed it. This
   * method provides a summary of all such removal events, and is used for
   * improved logging and debugging.
   * */
  std::string removalEventsStr() const { return removals.str(); }

protected:
  /**
   * Insert \a op into this Graph, and add it to the consumer lists of its
   * inputs' creators.
   * */
  OpId insertMultioutOp(std::unique_ptr<Op> op);
  OpId insertMultioutOp(const Op &op) {
    return insertMultioutOp(op.cloneMultioutOp());
  }

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
   * a lightweight record of it and its removal event are retained. This
   * record is particularly useful if there is an attempted access of
   * #opToRemove after it has been removed. Such an attempt will result in a
   * descriptive error, including the #removalContext string.
   * */
  void removeMultioutOp(OpId opToRemove,
                        const OptionalTensorIds &outputSubstitutes,
                        const std::string &removalContext);

  /** Methods to access Ops in this Graph as multiout::Ops. */
  Op &multioutOp(OpId id) { return op(id); }
  const Op &multioutOp(OpId id) const { return op(id); }

  OpId nxtOpId() const { return ops_.size(); }

  /**
   * Verify that this Graph is in a valid state, by checking the correctness
   * of the producer/consumer relationships between Ops.
   * */
  void assertMultioutGraphCorrectness() const;

private:
  /**
   * Derived classes must define what it means to be equivalent in this
   * virtual method.
   * */
  virtual bool multiOutTypeSpecificEqualTo(const Graph &) const = 0;

  /**
   * Return the  id'th Op in the member variable \a ops, performing checks
   * that 0 <= id < nOps().
   * */
  Op &op(OpId id);
  const Op &op(OpId) const;

  /** All of the Ops in this Graph. The Ops are stored as unique_ptrs wrapped
   * in the CopyByClone class, which makes them, and thus the class, copyable.
   * When this Graph is copied, the resulting copy has a clone of all of the
   * Ops in this Graph.
   */
  std::vector<poprithms::util::CopyByClone<Op>> ops_;

  /**
   * The Ops which have not been deleted.
   * */
  std::set<OpId> live_;

public:
  const RemovalEvents &removalEvents() const { return removals; }

private:
  /**
   * Every OpId in [0, ops_.size()) corresponds to either a 'live' Op, or to
   * an op which once was live, but has been removed. If it was removed, a
   * record of it and it's removal is kept. This object stores these records.
   * */
  RemovalEvents removals;

  // The name of this Graph.
  std::string name_;
};

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
