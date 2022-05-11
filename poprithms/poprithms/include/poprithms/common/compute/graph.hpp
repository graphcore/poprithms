// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_GRAPH
#define POPRITHMS_COMMON_COMPUTE_GRAPH

#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/common/compute/device.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/ndarray/deviceid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/tensorinfo.hpp>

namespace poprithms {
namespace common {
namespace compute {

using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OptionalTensorId;
using common::multiout::OptionalTensorIds;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using common::schedulable::SubGraphId;
using common::schedulable::SubGraphIds;
using poprithms::ndarray::DeviceId;
using poprithms::ndarray::DeviceIds;
using poprithms::ndarray::DType;
using poprithms::ndarray::DTypes;
using poprithms::ndarray::TensorInfo;
using poprithms::ndarray::TensorInfos;

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
   * Return true if the numerical type of tensor #tId is integral.
   * */
  bool isFixedPoint(const TensorId &) const;

  /**
   * The device of tensor #tId.
   * */
  DeviceId deviceId(const TensorId &tId) const;

  /**
   * The devices of the tensors #tIds.
   * */
  DeviceIds deviceIds(const TensorIds &tIds) const;

  /**
   * If all the tensors in #tIds are on the same device, then return that
   * device. Otherwise, throw an error.
   * */
  DeviceId deviceIdByConsensus(const TensorIds &tIds) const;

  /**
   * The devices of all of the input tensors of op #opId.
   * */
  DeviceIds inDeviceIds(OpId) const;

  /**
   * The devices of all of the outputs of op #opId.
   * */
  DeviceIds outDeviceIds(OpId) const;

  /**
   * The tensor information (shape, type, device) of the tensor #tId.
   * */
  TensorInfo tensorInfo(const TensorId &) const;

  /**
   * The tensor informations of the tensors #tIds.
   * */
  TensorInfos tensorInfos(const TensorIds &tIds) const;

  /**
   * the op with id #id.
   * */
  const Op &computeOp(OpId id) const;

  /**
   * Insert a new op into the graph, with inputs #ins in the sub-graph #sgId,
   * and outputs with shape, type and device defined by #outInfos. All
   * additional op attributes are matched to #args.
   * */
  template <class T, class... Args>
  OpId createComputeOp(const TensorIds &ins,
                       SubGraphId sgId,
                       const TensorInfos &outInfos,
                       Args... args);

  /**
   * The device which the tensor #tId is on.
   * */
  const Device &device(const TensorId &tid) const;

  /**
   * The type of the device that tensor #tId is on.
   * */
  DeviceType deviceType(const TensorId &tId) const;

  /**
   * The type of the device then inputs and outputs of the op #opId are on. If
   * either (1) not all inputs and outputs are on the same type of device or
   * (2) there are no inputs or outputs to op #opId, then an error is thrown.
   * */
  DeviceType deviceType(OpId opId) const;

  /**
   * The device types of each of the tensors in #tIds.
   * */
  DeviceTypes deviceTypes(const TensorIds &) const;

  /**
   * The types of the devices in #devIds.
   * */
  DeviceTypes deviceTypes(const DeviceIds &devIds) const;

  /**
   * The type of the device #dId.
   * */
  DeviceType deviceType(DeviceId) const;

  /**
   * The replication factor of all of the ipu tensors in this graph.
   * */
  virtual uint64_t replicationFactor_u64() const = 0;

  /**
   * The device with id #dId.
   * */
  virtual const Device &device(DeviceId dId) const = 0;

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

  /**
   * Verify that the attributes of a single op are valid.
   * */
  virtual void verifyComputeDerivedOpValid(OpId) const = 0;
  void verifySchedulableDerivedOpValid(OpId) const final;
  void verifyValidAtComputeLevel(OpId) const;

  /**
   * Verify that the attributes of the entire graph are valid.
   * */
  virtual void verifyComputeDerivedGraphValid() const = 0;
  void verifySchedulableDerivedGraphValid() const final;

  // These's methods are protected in the parent class, but should not be used
  // beyond this class and are therefore made private:
  using schedulable::Graph::insertSchedulableOp;
  using schedulable::Graph::schedulableTypeSpecificRemoveOp;
  using schedulable::Graph::schedulableTypeSpecificVerifyValidSubstitute;
};

template <class T, class... Args>
OpId Graph::createComputeOp(const TensorIds &inIds,
                            SubGraphId sgId,
                            const TensorInfos &outs,
                            Args... args) {

  if (!inIds.empty() && subGraphIdFromTensorIds(inIds) != sgId) {
    std::ostringstream oss;
    oss << "Inputs " << inIds << " not in sub-graph " << sgId;
    throw poprithms::error::error("common::compute", oss.str());
  }

  auto state =
      Op::State::getStartingState(nxtOpId(), sgId, inIds, outs, *this);

  auto opId = insertComputeOp(std::unique_ptr<T>(new T(state, args...)));
  return opId;
}

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
