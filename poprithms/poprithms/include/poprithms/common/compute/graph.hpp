// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_GRAPH
#define POPRITHMS_COMMON_COMPUTE_GRAPH

#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/ndarray/dtype.hpp>

namespace poprithms {
namespace common {
namespace compute {

class Op;

using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OptionalTensorId;
using common::multiout::OptionalTensorIds;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using poprithms::ndarray::DType;
using poprithms::ndarray::DTypes;

/**
 * A graph class combining multiple poprithms components -- autodiff,
 * unwinding, alias analysis, & host tensors.
 * */
class Graph : public poprithms::common::schedulable::Graph {

public:
  Graph()              = default;
  Graph(Graph &&)      = default;
  Graph(const Graph &) = default;
  Graph &operator=(Graph &&) = default;
  Graph &operator=(const Graph &) = default;
  virtual ~Graph() override       = default;

  /**
   * The numerical type of the elements of tensor #tId.
   * */
  DType dtype(const TensorId &tId) const;

  /**
   * The numerical types of the elements of tensor #tId.
   * */
  DTypes dtypes(const TensorId &) const;

  /**
   * the op with id #id.
   * */
  const Op &computeOp(OpId id) const;

protected:
  OpId insertComputeOp(std::unique_ptr<Op>);

private:
  /**
   * the op with id #id (const and non-const versions). Design note: splitting
   * the const and non-const versions across access scopes (i.e. making the
   * const version public) makes use by non-const objects messy, hence the
   * public version with a different name (computeOp).
   * */
  const Op &op(OpId) const;
  Op &op(OpId);

  // These's methods are protected in the parent class, but should not be used
  // beyond this class and are therefore made private:
  using schedulable::Graph::insertSchedulableOp;
  using schedulable::Graph::schedulableTypeSpecificRemoveOp;
  using schedulable::Graph::schedulableTypeSpecificVerifyValidSubstitute;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
