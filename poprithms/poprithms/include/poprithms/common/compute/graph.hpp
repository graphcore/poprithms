// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_GRAPH
#define POPRITHMS_COMMON_COMPUTE_GRAPH

#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/common/compute/host.hpp>
#include <poprithms/common/compute/ipu.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/remote.hpp>
#include <poprithms/common/compute/replication.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/deviceid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/tensorinfo.hpp>
#include <poprithms/program/distributed/codelocation.hpp>
#include <poprithms/util/copybyclone.hpp>
#include <poprithms/util/stringutil.hpp>

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
using ndarray::DeviceId;
using ndarray::DeviceIds;
using ndarray::DType;
using ndarray::DTypes;
using ndarray::TensorInfo;
using ndarray::TensorInfos;
using poprithms::common::multiout::OpTraversal;
using poprithms::common::multiout::OpTraversals;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using program::distributed::CodeLocation;

/**
 * A graph class combining multiple poprithms components -- autodiff,
 * unwinding, alias analysis, & host tensors.
 * */
class Graph : public poprithms::common::schedulable::Graph {

public:
  Graph(Graph &&);
  Graph(const Graph &);

  Graph &operator=(Graph &&);
  Graph &operator=(const Graph &);

  /**
   * Create a graph containing an ipu with #nTilesPerReplica tiles per
   * replica, and #rf replicas.
   * */
  Graph(uint64_t nTilesPerReplica, ReplicationFactor);

  Graph() : Graph(32, ReplicationFactor::create(1)) {}

  virtual ~Graph() override;

  /**
   * The numerical type of the elements of tensor #tId.
   * */
  DType dtype(const TensorId &tId) const;

  /**
   * The number of bytes (number_of_elements * bytes_per_element) of tensor
   * #tId.
   * */
  uint64_t nbytes(const TensorId &tId) const;

  /**
   * The numerical types of the elements of tensors #tIds.
   * */
  DTypes dtypes(const TensorIds &tIds) const;

  /**
   * \return true if the numerical type of tensor #tId is integral.
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
   * If all the tensors in #tIds are on the same device, then return the
   * device id. If not, throw an error.
   * */
  DeviceId deviceIdByUnanimity(const TensorIds &tIds) const;

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
   * Dynamically cast op #opId to type OP.
   * */
  template <typename OP> const OP *dynamicCast(OpId) const;
  template <typename OP> OP *dynamicMutableCast(OpId);

  /**
   * Dynamically cast op #opId to type OP, and throw an error if this fails.
   * */
  template <typename OP> const OP *castOrThrow(OpId) const;
  template <typename OP> OP *mutableCastOrThrow(OpId);

  /**
   * All ops in the sub-graph #sgId which can be cast to type T0.
   * */
  template <typename T0> OpIds opIds(SubGraphId) const;

  /**
   * All ops (in all sub-graphs) which can be cast to type T0.
   * */
  template <typename T0> OpIds opIds() const;

  /**
   * The device which the tensor #tId is on.
   * */
  const Device &device(const TensorId &tid) const;

  /**
   * The type of the device that tensor #tId is on.
   * */
  DeviceType deviceType(const TensorId &tId) const;

  /**
   * The type of the device that inputs and outputs of the op #opId are on. If
   * either (1) not all inputs and outputs are on the same type of device or
   * (2) there are no inputs or outputs of op #opId, then an error is thrown.
   * */
  DeviceType deviceTypeByUnanimity(OpId opId) const;

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
   * Set the initial value of the ipu tensor #tId on replica #r to #initVal.
   * */
  void
  setInitialValue(const TensorId &tId, uint64_t r, const HostTensor &initVal);

  /**
   * The sub-graph of op #opId.
   * */
  SubGraphId subGraphId(OpId) const;

  /**
   * The sub-graph of tensor #tId.
   * */
  SubGraphId subGraphId(const TensorId &tId) const;

  /**
   * \return true if the tensor #tId is a remote device.
   * */
  bool isOnRemote(const TensorId &) const;

  /**
   * \return true if the tensor #tId is on the host device.
   * */
  bool isOnHost(const TensorId &) const;

  /**
   * \return true if the tensor #tId is on an ipu device.
   * */
  bool isOnIpu(const TensorId &) const;

  /**
   * The total number of devices.
   * */
  uint64_t nDevices() const { return devices.size(); }

  /**
   * Check that the tensor #tId is a host tensor. If it is not, throw a
   * descriptive error.
   * */
  void verifyIsHost(const TensorId &) const;

  /**
   * Check that the tensor #tId is a remote tensor. If it is not, throw a
   * descriptive error.
   * */
  void verifyIsRemote(const TensorId &) const;

  /**
   * Check that the tensor #tId is an ipu tensor. If it is not, throw a
   * descriptive error.
   * */
  void verifyIsIpu(const TensorId &) const;

  /**
   * Check that device #devId is an ipu device.
   * */
  void verifyIsIpu(const DeviceId &) const;

  /**
   * All ipu devices.
   * */
  DeviceIds ipuDevices() const;

  /**
   * All ipu devices, other than the root ipu.
   * */
  DeviceIds nonRootIpuDevices() const;

  /**
   * All remote devices.
   * */
  DeviceIds remoteDevices() const;

  /**
   * The columns of the attributes specific to common::compute::Ops.
   * */
  std::vector<poprithms::util::StringColumn>
  getComputeColumns(const OpIds &,
                    const poprithms::util::StringColumn::Parameters &) const;

  void appendOpColumns(std::ostream &, const OpIds &) const final;

  bool hasDerivedRefs(const TensorId &tId) const {
    return op(tId.opId()).hasDerivedRefs(tId.outIndex());
  }

  bool isRootRef(const TensorId &tId) const {
    return op(tId.opId()).isRootRef(tId.outIndex());
  }

  TensorId rootRef(const TensorId &tId) const {
    return op(tId.opId()).rootRef(tId.outIndex());
  }

  TensorIds refsExcludingSelf(const TensorId &tId) const {
    return op(tId.opId()).refsExcludingSelf(tId.outIndex());
  }

  /**
   * All tensors which reference a tensor in a different sub-graph (root or
   * derived).
   * */
  TensorIds tensorsWithRefs() const;

  /**
   * All root references. Specifically, all tensors which have derived
   * references in different sub-graphs.
   * */
  TensorIds rootRefs() const;

  /**
   * All derived reference tensors. Specifically, all tensors whose root
   * tensor is in a different sub-graph.
   * */
  TensorIds derivedRefs() const;

  /**
   * The device id of the host device.
   * */
  DeviceId host() const { return DeviceId(0); }

  /**
   * The device id of the 'root ipu'. This is the ipu with all available
   * tiles, and it corresonds to the top-level poplar graph from which virtual
   * graphs are created.
   * */
  DeviceId rootIpu() const { return DeviceId(1); }

  uint64_t nTilesPerReplica() const { return nTilesPerReplica_; }

  uint64_t replicationFactor_u64() const {
    return replicationFactor_.get_u64();
  }

  const Device &device(DeviceId id) const;

  /**
   * The total number of ipu tiles, across all replicas.
   * */
  uint64_t nTiles() const {
    return nTilesPerReplica() * replicationFactor_.get_u64();
  }

  /**
   * Return an ipu with a subset of #ipu0's tiles. Specifically, return a
   * device with the (ranked) tiles from #rank0 to #rank1 of ipu0.
   *
   * Example: if #ipu0 has tiles made up of the intervals [2,4) and [6,9),
   *          and if rank0 = 1 and rank1 = 4:
   *
   *    0 1 2 3 4 5 6 7 8 9    : all tiles
   *        [   )   [     )
   *    . . x x . . x x x .    : tiles of ipu0 ('x' = included)
   *        0 1     2 3 4      : indices of tiles of ipu0
   *          ^         ^
   *          |         |
   *        rank0      rank1
   *
   * then the subset of tiles is [1,2), [6,8).
   *
   * The returned device has rank1 - rank0 tiles.
   *
   * \sa Intevals::subIntervals.
   * */
  DeviceId ipu(DeviceId ipu0, uint64_t rank0, uint64_t rank1);

  /**
   * \return N ipu devices each with 1/N of the tiles of #ipu0.
   * */
  std::vector<DeviceId> partition(DeviceId, uint64_t N);

  /**
   * The ipu device with id #dId. If #dId is not an ipu, an error is thrown.
   * */
  const Ipu &ipu(DeviceId) const;

  /**
   * The remote device with id #dId. If #dId is not a remote device, an error
   * is thrown.
   * */
  const Remote &remote(DeviceId) const;

  /**
   * Map from one enum type to the other.
   *
   * DeviceType::Ipu    -> CodeLocation::Ipu
   * DeviceType::Host   -> CodeLocation::Host
   * DeviceType::Remote -> invalid (no code).
   * */
  static CodeLocation codeLocationFromDeviceType(DeviceType);

  CodeLocation codeLocation(OpId id) const { return op(id).codeLocation(); }

  bool isPartiallyHost(OpId id) const { return op(id).isPartiallyHost(); }

  std::string str(OpId id) const { return op(id).str(); }

  /**
   * Specify that the sub-graphs #sgIds are runnable. That is, they are entry
   * points of execution of the graph. These are analagous to the set of
   * programs passed to a poplar engine. Any sub-graph can be made runnable.
   * */
  void setRunnable(const std::vector<SubGraphId> &sgIds);

  /**
   * \return true if #sgId is a runnable sub-graph, set with the method
   *         #setRunnable.
   * */
  bool isRunnable(SubGraphId sgId) const;

  std::vector<SubGraphId> runnable() const { return runnable_; }

  using poprithms::common::multiout::Graph::opIds;
  using poprithms::common::schedulable::Graph::opIds;

  /**
   * All the tensors which are on the host device.
   * */
  TensorIds hostTensors() const;

  /**
   * The DAG consisting of all caller->callee edges. Specifically, if an op in
   * graph #g0 has a callee sub-graph g1, then there is an edge #g0->#g1, and
   * the returned vector #edges has #g1 in #edges[g0].
   * */
  std::vector<std::vector<uint64_t>> calleeGraph() const;

  /**
   * For all sub-graphs which are callees, return the set of ops call into
   * them, and their context. Specifically, return a map whose keys are the
   * sub-graphs, and the values are the call events where the callee is the
   * map key.
   * */
  std::map<SubGraphId, CallEvents> callEvents() const;

  SubGraphIds callees(OpId id) const { return op(id).callees(); }

  uint64_t nCallees(OpId id) const { return op(id).nCallees(); }

  /**
   * For ops with unique callees, return the unique CallEvent. If #opId has
   * multiple callees, this method throws an error.
   * */
  CallEvent callEvent(OpId opId) const;

  /**
   * Starting from the sub-graphs in #sgIds and traversing the DAG formed by
   * edges between callers and callees (see #calleeGraph), traverse to all
   * reachable sub-graphs.
   * */
  SubGraphIds reachable(const SubGraphIds &) const;

  SubGraphIds reachableFromRunnable() const { return reachable(runnable()); }

  /**
   * \return all ops with one or more callees.
   * */
  OpIds opsWithCallees() const;

  bool atLeastOneOutIsIpu(OpId id) const {
    return op(id).atLeastOneOutIsIpu();
  }

  /**
   * If the tensor #tId is
   * 1) in a sub-graph which is the callee of a calling op,
   * 2) is the destination of a copy into the callee from the calling
   *    sub-graph,
   * then the calling op and index of the copy are an element of the returned
   * vector.
   *
   * \sa Op::inCopies.
   * */
  std::vector<std::pair<CallEvent, InIndex>>
  indexedInCopies(const TensorId &tId) const;

  /**
   * \sa indexedInCopies and Op::outCopies. This method returns all copies out
   * from the callee sub-graphs into the calling sub-graph.
   * */
  std::vector<std::pair<CallEvent, OutIndex>>
  indexedOutCopies(const TensorId &) const;

  /**
   * A string summarizing the ops in #opIds.
   * */
  std::string str(const OpIds &opIds) const;

  /**
   * Create a clone of op #opId in the sub-graph #sgId, with input tensors
   * #inTensors. The new input tensors #inTensors must have the same type and
   * shape as #opId, and they must be in the sub-graph #sgId.
   *
   * Topological constraints are not transferred when cloning with this
   * method.
   * */
  OpId clone(OpId opId, const TensorIds &inTensors, SubGraphId sgId);

  /**
   * The source of the copy to the callee tensor #inCallee for the call event
   * #ce. The returned tensor is in the calling sub-graph.
   * */
  TensorId srcInCaller(const TensorId &inCallee, const CallEvent &ce) const;

  /**
   * The destination of the copy from the callee tensor #inCallee, for the
   * call event #ce. This copy happens at the end of the call event, when the
   * tensor in the callee sub-graph is copied to a tensor in the calling
   * sub-graph.
   * */
  TensorId dstInCaller(const TensorId &inCallee, const CallEvent &ce) const;

  /**
   * For an op #opId with just 1 callee sub-graph, this method returns the
   * destination in the calling sub-graph (the sub-graph containing #opId) of
   * the copy at the end of the call event from the callee tensor #inCallee.
   * */
  TensorId dstInCaller(const TensorId &inCallee, OpId opId) const {
    return dstInCaller(inCallee, callEvent(opId));
  }

  /**
   * \return true if the call event #ce has a tensor copied out at index #o.
   * */
  bool hasSrcInCallee(const CallEvent &, OutIndex o) const;

  /**
   * The tensor copied out of the callee sub-graph of #ce, at output index #o.
   * */
  TensorId srcInCallee(const CallEvent &ce, OutIndex o) const;

  /**
   * The destinations in a callee sub-graph to which the tensor #inCaller is
   * copied at the start of the call event #ce.
   * */
  TensorIds dstsInCallee(const TensorId &inCaller, const CallEvent &ce) const;

  /**
   * \return true if the tensor #inCallee is copied to from a tensor in the
   *         calling sub-graph in the call event #ce.
   * */
  bool isDstInCallee(const TensorId &inCallee, const CallEvent &ce) const;

  /**
   * \return true is the tensor #inCallee is copied from in the call event
   * #ce. The destination of such a copy is a tensor in calling sub-graph.
   * */
  bool isSrcInCallee(const TensorId &inCallee, const CallEvent &) const;

  /**
   * This method checks true if a non-zero gradient propagate across in the
   * input-output indices of #ot.
   *
   * Specifically, this method returns true if (1) the input and output
   * tensors of #ot are floating point and (2) the op of #ot can propagate the
   * gradients (\sa Op::gradientPropagates).
   *
   * */
  bool gradientPropagates(const OpTraversal &ot) const;

  /**
   * Recall that a TensorId is a pair containing (1) the id of the op which
   * creates the tensor and (2) the output index of the tensor. An OpTraversal
   * is a triplet, which adds to (1) and (2) and input index.
   *
   * This method checks if there are any input indices for the for the op of
   * #tId for which the corresponding OpTraversal can propagate a gradient.
   * */
  bool gradientPropagates(const TensorId &tId) const;

  /**
   * Tensors which are streamed between an ipu device and the host have a
   * particular shape relationship, in part due to the implicit replication of
   * tensors on ipu. Specifically, cpu tensors have 2 more dimensions than
   * their ipu counterparts. These 2 dimensions, which appear before the
   * other, normal, 'shape' dimensions are the parameters.
   *
   *  \param fanFactor The size of the circular buffer, or the number of times
   *                   the tensor is copied between host and device before the
   *                   buffer wraps around to the starting position.
   *
   * \param replicationFactor The replication factor of the ipu. Host tensors
   *                           have explicit replication dimensions.
   *
   * \param ipuShape The shape of the tensor on the ipu.
   *
   * \return  (fanFactor, replicationFactor, *ipuShape).
   *
   * */
  static Shape getHostShape(CircularBufferCount fanFactor,
                            ReplicationFactor,
                            const Shape &ipuShape);

  /**
   * If all tensors in the sub-graph #sgId and its callees (recursively) are
   * on the same device, return that device. If not, error.
   * */
  DeviceId deviceId(SubGraphId) const;

  /**
   * \return true if the op #opId initializes a constant tensor.
   * */
  virtual bool isConstInit(OpId) const = 0;

  /**
   * \return The constant value that the op #opId initializes.
   * */
  virtual HostTensor constInitValue(OpId) const = 0;

  /**
   * \return true if the op #opId initializes a variable (=non-constant)
   * tensor.
   * */
  virtual bool isVarInit(OpId) const = 0;

protected:
  OpId insertComputeOp(std::unique_ptr<Op>);

  bool computeTypeSpecificEqualTo(const Graph &rhs) const;

  /**
   * Insert an op of type TRefFromOp, which has zero inputs and have #srcId as
   * an attribute. This is a special kind of initializer (input) op, which
   * does not create a new variable/constant but rather just aliases a
   * variable in a different graph. See the Op class for further information.
   * */
  template <class TRefFromOp>
  TensorId tRefFrom(const TensorId &srcId, SubGraphId destination);

  /**
   * Create a remote device associated to the ipu #ipu, of numerical type
   * #dtype.
   * */
  DeviceId createRemote(DeviceId ipu,
                        DType dtype,
                        const Shape &,
                        const RemoteOptions &);

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
  void verifyValidFromComputeLevel(OpId) const;

  /**
   * Verify that the attributes of the entire graph are valid.
   * */
  virtual void verifyComputeDerivedGraphValid() const = 0;
  void verifySchedulableDerivedGraphValid() const final;

  void schedulableTypeSpecificRemoveOp(
      OpId opToRemove,
      const OptionalTensorIds &outputSubstitutes) final;

  void schedulableTypeSpecificVerifyValidSubstitute(
      const TensorId &before,
      const TensorId &after) const final;

  // These's methods are protected in the parent class, but should not be used
  // beyond this class and are therefore made private:
  using schedulable::Graph::insertSchedulableOp;
  using schedulable::Graph::schedulableTypeSpecificRemoveOp;
  using schedulable::Graph::schedulableTypeSpecificVerifyValidSubstitute;

  std::vector<poprithms::util::CopyByClone<Device>> devices;
  uint64_t nTilesPerReplica_;
  ReplicationFactor replicationFactor_;
  SubGraphIds runnable_;
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
template <class T>
TensorId Graph::tRefFrom(const TensorId &srcId,
                         const SubGraphId destination) {

  /// Obtain the canonical representative of #srcId.
  const TensorId rootId = op(srcId.opId()).rootRef(srcId.outIndex());

  /// If #rootId is already in the sub-graph #destination, then do not create
  /// a new op, just return #rootId.
  if (subGraphId(rootId) == destination) {
    return rootId;
  }

  /// If there is already a reference to #rootId in #destination, do not
  /// create a new op, rather re-use the existing one.
  for (auto existingDerived :
       op(rootId.opId()).derivedRefs(rootId.outIndex())) {
    if (subGraphId(existingDerived) == destination) {
      return existingDerived;
    }
  }

  const auto opId =
      createComputeOp<T>({}, destination, {tensorInfo(srcId)}, rootId);

  const TensorId dst = op(opId).outTensorId(OutIndex(0));

  op(rootId.opId()).insertOutDerivedRef(rootId.outIndex(), dst);

  return dst;
}

template <typename T0> OpIds Graph::opIds(SubGraphId subGraphId) const {
  OpIds ids;
  for (auto opId : poprithms::common::schedulable::Graph::opIds(subGraphId)) {
    if (dynamicCast<T0>(opId)) {
      ids.push_back(opId);
    }
  }
  return ids;
}

template <typename T0> OpIds Graph::opIds() const {
  OpIds ids;
  for (auto opId : opIds()) {
    if (dynamicCast<T0>(opId)) {
      ids.push_back(opId);
    }
  }
  return ids;
}

template <typename T> const T *Graph::dynamicCast(OpId opId) const {
  return dynamic_cast<const T *>(&op(opId));
}

template <typename T> const T *Graph::castOrThrow(OpId opId) const {
  auto cst = dynamic_cast<const T *>(&op(opId));
  if (!cst) {
    std::ostringstream oss;
    oss << "Failed to cast op " << op(opId)
        << " to type with typeid name:" << typeid(T).name();
    throw poprithms::error::error("common::compute", oss.str());
  }
  return cst;
}

template <typename T> T *Graph::mutableCastOrThrow(OpId opId) {
  auto cst = dynamic_cast<T *>(&op(opId));
  if (!cst) {
    std::ostringstream oss;
    oss << "Failed to cast op " << op(opId)
        << " to type with typeid name:" << typeid(T).name();
    throw poprithms::error::error("common::compute", oss.str());
  }
  return cst;
}

template <typename T> T *Graph::dynamicMutableCast(OpId opId) {
  return dynamic_cast<T *>(&op(opId));
}

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
