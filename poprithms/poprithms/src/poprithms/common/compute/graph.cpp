// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iterator>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/host.hpp>
#include <poprithms/common/compute/ipu.hpp>
#include <poprithms/common/compute/memoryaliasmapper.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/nop.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/remote.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/unisort.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace {
std::string getDeviceAssertionMessage(const std::string &device,
                                      const Graph &m,
                                      const TensorId &id) {

  std::ostringstream oss;
  oss << "Failed to assert that the tensor " << id << " is a " << device
      << " tensor. This tensor has tensor info " << m.tensorInfo(id)
      << ", and is created by the op " << m.computeOp(id.opId()) << ". ";
  return oss.str();
}
} // namespace

void Graph::verifyIsHost(const TensorId &id) const {
  if (!isOnHost(id)) {
    throw error(getDeviceAssertionMessage("host", *this, id));
  }
}

bool Graph::computeTypeSpecificEqualTo(const Graph &r) const {
  return devices == r.devices && nTilesPerReplica_ == r.nTilesPerReplica_ &&
         replicationFactor_ == r.replicationFactor_ &&
         runnable_ == r.runnable_;
}

void Graph::verifyIsRemote(const TensorId &tId) const {
  if (!isOnRemote(tId)) {
    throw error(getDeviceAssertionMessage("remote", *this, tId));
  }
}

void Graph::verifyIsIpu(const DeviceId &dId) const {
  if (!device(dId).isIpu()) {
    std::ostringstream oss;
    oss << "Failed in verifyIsIpu(DeviceId = " << dId << "). This device is "
        << device(dId);
    throw error(oss.str());
  }
}

void Graph::verifyIsIpu(const TensorId &tId) const {
  if (!isOnIpu(tId)) {
    throw error(getDeviceAssertionMessage("ipu", *this, tId));
  }
}

TensorIds Graph::rootRefs() const {
  TensorIds tids;
  for (auto opId : opIds()) {
    const auto &op_ = op(opId);
    for (OutIndex o = 0; o < nOutTensors(opId); ++o) {
      if (op_.nDerivedRefs(o) != 0) {
        tids.push_back({opId, o});
      }
    }
  }
  return tids;
}

TensorIds Graph::derivedRefs() const {
  TensorIds tids;
  for (auto opId : opIds()) {
    const auto &op_ = op(opId);
    for (OutIndex o = 0; o < nOutTensors(opId); ++o) {
      if (!op_.isRootRef(o)) {
        tids.push_back({opId, o});
      }
    }
  }
  return tids;
}

OpId Graph::clone(OpId opId, const TensorIds &inIds, SubGraphId sgId) {

  verifySubGraphId(inIds, sgId);

  if (op(opId).inShapes() != shapes(inIds)) {
    std::ostringstream oss;
    oss << "The shapes of the inputs to the op being cloned, " << op(opId)
        << ", are ";
    poprithms::util::append(oss, op(opId).inShapes());
    oss << ". The shapes of the new inputs are ";
    poprithms::util::append(oss, shapes(inIds));
    oss << ". They should be the same (shape inference is not rerun).";
    throw error(oss.str());
  }

  if (dtypes(inTensorIds(opId)) != dtypes(inIds)) {
    std::ostringstream oss;
    oss << "The dtypes of the inputs to the op being cloned, " << op(opId)
        << ", are ";
    poprithms::util::append(oss, dtypes(inTensorIds(opId)));
    oss << ". The dtypes of the new inputs are ";
    poprithms::util::append(oss, shapes(inIds));
    oss << ". They should be the same (dtype inference is not rerun).";
    throw error(oss.str());
  }

  // note that topo-cons are not copied across.
  const auto state = Op::State::getStartingState(
      OpId(nxtOpId()), sgId, inIds, tensorInfos(outTensorIds(opId)), *this);

  auto foo = op(opId).cloneWithState(state);

  return insertComputeOp(std::move(foo));
}

TensorId Graph::srcInCaller(const TensorId &inCallee,
                            const CallEvent &cse) const {
  const auto &op_ = op(cse.caller());
  return op_.inTensorId(op_.inIndex({inCallee, cse.index()}));
}

TensorId Graph::dstInCaller(const TensorId &inCallee,
                            const CallEvent &ce) const {
  auto outIndex =
      op(ce.caller()).outIndex(CalleeTensorId(inCallee, ce.index()));
  return op(ce.caller()).outTensorId(outIndex);
}

bool Graph::isDstInCallee(const TensorId &inCallee,
                          const CallEvent &cse) const {
  return op(cse.caller()).isDstInCallee({inCallee, cse.index()});
}

bool Graph::isSrcInCallee(const TensorId &inCallee,
                          const CallEvent &cse) const {
  return op(cse.caller()).isSrcInCallee({inCallee, cse.index()});
}

TensorIds Graph::dstsInCallee(const TensorId &inCaller,
                              const CallEvent &ce) const {
  return op(ce.caller()).dstsInCallee({inCaller, ce.index()});
}

bool Graph::hasSrcInCallee(const CallEvent &cse, OutIndex o) const {
  return op(cse.caller()).isCopiedOut(o, cse.index());
}

TensorId Graph::srcInCallee(const CallEvent &cse, OutIndex o) const {
  return op(cse.caller()).srcInCallee(o, cse.index());
}

TensorIds Graph::tensorsWithRefs() const {
  TensorIds tids;
  for (auto opId : opIds()) {
    const auto &op_ = op(opId);
    for (OutIndex o = 0; o < nOutTensors(opId); ++o) {
      if (!op_.refsExcludingSelf(o).empty()) {
        tids.push_back({opId, o});
      }
    }
  }
  return tids;
}

bool Graph::isOnHost(const TensorId &tId) const {
  return deviceType(tId) == DeviceType::Host;
}

bool Graph::isOnIpu(const TensorId &tId) const {
  return deviceType(tId) == DeviceType::Ipu;
}

bool Graph::isOnRemote(const TensorId &tId) const {
  return deviceType(tId) == DeviceType::Remote;
}

SubGraphId Graph::subGraphId(OpId id) const { return op(id).subGraphId(); }

SubGraphId Graph::subGraphId(const TensorId &tId) const {
  return subGraphId(tId.opId());
}

Graph::Graph(Graph &&)                 = default;
Graph::Graph(const Graph &)            = default;
Graph &Graph::operator=(Graph &&)      = default;
Graph &Graph::operator=(const Graph &) = default;
Graph::~Graph()                        = default;

Graph::Graph(uint64_t nTilesPerReplica, ReplicationFactor r)
    : nTilesPerReplica_(nTilesPerReplica), replicationFactor_(r) {

  // 2 devices are created automatically for a graph:
  //   Host device (DeviceId=0)
  //   Root ipu device (DeviceId=1)

  DeviceId hostId = devices.size();
  devices.emplace_back(std::make_unique<Host>(hostId));

  DeviceId rootIpuId = devices.size();
  devices.emplace_back(std::make_unique<Ipu>(
      rootIpuId, poprithms::util::Intervals(0, nTilesPerReplica)));

  if (devices.at(host().get_u64()).uptr->deviceType() != DeviceType::Host) {
    std::ostringstream oss;
    oss << "The DeviceId of the host should be " << host()
        << ", but a check in the constructor failed to assert this. ";
    throw error(oss.str());
  }

  if (devices.at(rootIpu().get_u64()).uptr->deviceType() != DeviceType::Ipu) {
    std::ostringstream oss;
    oss << "The DeviceId of the root ipu should be " << host()
        << ", but a check in the constructor failed to assert this. ";
    throw error(oss.str());
  }
}

void Graph::schedulableTypeSpecificRemoveOp(
    OpId opToRemove,
    const OptionalTensorIds &outputSubstitutes) {

  const auto &op_ = op(opToRemove);

  for (OutIndex o = 0; o < op_.nOutTensors(); ++o) {
    if (!op_.isRootRef(o)) {
      const auto root = op_.rootRef(o);
      op(root.opId()).removeOutDerivedRef(root.outIndex(), {opToRemove, o});
    }
  }

  (void)opToRemove;
  (void)outputSubstitutes;
}

void Graph::schedulableTypeSpecificVerifyValidSubstitute(
    const TensorId &before,
    const TensorId &after) const {
  if (deviceId(before) != deviceId(after)) {
    std::ostringstream oss;
    oss << "DeviceId of tensor before substitution is " << deviceId(before)
        << ", and DeviceId after substitution is " << deviceId(after) << ". ";
    throw error(oss.str());
  }

  if (dtype(before) != dtype(after)) {
    std::ostringstream oss;
    oss << "Type of tensor before substitution is " << dtype(before)
        << ", and type after substitution is " << deviceId(after) << ". ";
    throw error(oss.str());
  }
}

DeviceType Graph::deviceType(const TensorId &tensorId) const {
  return device(tensorId).deviceType();
}
CodeLocation Graph::codeLocationFromDeviceType(DeviceType dt) {
  switch (dt) {
  case DeviceType::Host: {
    return CodeLocation::Host;
  }
  case DeviceType::Ipu: {
    return CodeLocation::Ipu;
  }
  default: {
    std::ostringstream oss;
    oss << "No CodeLocation equivalent for " << dt << '.';
    throw error(oss.str());
  }
  }
}

DeviceType Graph::deviceTypeByUnanimity(OpId opId) const {
  return op(opId).deviceTypeByUnanimity();
}
DeviceType Graph::deviceType(DeviceId devId) const {
  return device(devId).deviceType();
}

std::vector<DeviceType> Graph::deviceTypes(const TensorIds &ids) const {
  std::vector<DeviceType> ts;
  ts.reserve(ids.size());
  for (auto id : ids) {
    ts.push_back(deviceType(id));
  }
  return ts;
}

const Device &Graph::device(const TensorId &tid) const {
  return device(op(tid.opId()).outDeviceId(tid.outIndex()));
}

OpId Graph::insertComputeOp(std::unique_ptr<Op> nxtOp) {
  const auto newId = insertSchedulableOp(std::move(nxtOp));

  // Verify that the attributes at this level of abstraction, and at all
  // derived levels, are valid.
  verifyValidFromComputeLevel(newId);
  return newId;
}

void Graph::verifyValidAtComputeLevel(OpId opId) const {
  op(opId).verifyValidAtComputeLevel();
}

void Graph::verifyValidFromComputeLevel(OpId opId) const {
  op(opId).verifyValidFromComputeLevel();
}

void Graph::verifySchedulableDerivedGraphValid() const {
  for (auto opId : multiout::Graph::opIds()) {
    verifySchedulableDerivedOpValid(opId);
  }
}

void Graph::verifySchedulableDerivedOpValid(OpId opId) const {
  verifyValidFromComputeLevel(opId);
}

DType Graph::dtype(const TensorId &tId) const {
  auto t = computeOp(tId.opId()).outDType(tId.outIndex());
  return t;
}

const Op &Graph::computeOp(OpId a) const {
  // We know that all Ops in this Graph can be safely cast, so no need for
  // dynamic_cast here.
  return static_cast<const Op &>(multioutOp(a));
}

const Op &Graph::op(OpId a) const { return computeOp(a); }

// See Scott Meyers' "Effective C++"
Op &Graph::op(OpId id) {
  return const_cast<Op &>(static_cast<const Graph &>(*this).op(id));
}

DeviceIds Graph::deviceIds(const TensorIds &tIds) const {
  DeviceIds devIds;
  for (auto tId : tIds) {
    devIds.push_back(deviceId(tId));
  }
  return devIds;
}

DeviceId Graph::deviceIdByUnanimity(const TensorIds &all) const {

  if (all.empty()) {
    std::ostringstream oss;
    oss << "Unable to determine what the DeviceId is from an empty set of "
           "tensor ids.";
    throw error(oss.str());
  }

  const auto dIds = deviceIds(all);

  for (auto devId : dIds) {
    if (devId != dIds[0]) {
      std::ostringstream oss;
      oss << "Failed to determine a DeviceId by unanimity "
          << ", as not all tensors have the same DeviceId. "
          << "The DeviceIds were " << dIds << ", corresponding to tensors "
          << all << ", which are in sub-graphs " << subGraphIds(all) << '.';
      throw error(oss.str());
    }
  }

  return dIds[0];
}

bool Graph::isFixedPoint(const TensorId &tId) const {
  return poprithms::ndarray::isFixedPoint(dtype(tId));
}

DeviceId Graph::deviceId(const TensorId &tId) const {
  return op(tId.opId()).outDeviceId(tId.outIndex());
}

DeviceIds Graph::inDeviceIds(OpId opId) const {
  return op(opId).inDeviceIds();
}

DeviceIds Graph::outDeviceIds(OpId opId) const {
  return op(opId).outDeviceIds();
}
TensorInfo Graph::tensorInfo(const TensorId &tId) const {
  return op(tId.opId()).outTensorInfo(tId.outIndex());
}

TensorInfos Graph::tensorInfos(const TensorIds &ids) const {
  std::vector<TensorInfo> infos;
  infos.reserve(ids.size());
  for (const auto &id : ids) {
    infos.push_back(tensorInfo(id));
  }
  return infos;
}
namespace {
template <typename T> std::string getStr(const std::vector<T> &X) {
  std::ostringstream ost;
  poprithms::util::append(ost, X);
  return ost.str();
}
} // namespace

void Graph::appendOpColumns(std::ostream &ost, const OpIds &opIds_) const {

  const auto colParams = poprithms::util::StringColumn::Parameters()
                             .abridgeToSingleRow(false)
                             .thresholdWidth(50);

  auto cols = getMultioutColumns(opIds_, colParams);

  for (auto &&c : getSchedulableColumns(opIds_, colParams)) {
    cols.push_back(c);
  }

  for (auto &&c : getComputeColumns(opIds_, colParams)) {
    cols.push_back(c);
  }

  ost << alignedColumns(cols);
}

std::vector<poprithms::util::StringColumn> Graph::getComputeColumns(
    const OpIds &opIds_,
    const poprithms::util::StringColumn::Parameters &colParams) const {

  std::vector<poprithms::util::StringColumn> cols;

  const auto nRows = nMultioutRows(opIds_);
  using Strings    = std::vector<std::string>;

  Strings deviceStrings(nRows, "");
  Strings dtypeStrings(nRows, "");
  Strings rootRefOf(nRows, "");
  bool hasOutRefs{false};

  uint64_t rowIndex{0};
  for (auto opId : opIds_) {

    const auto &op_       = op(opId);
    const auto subGraphId = op_.subGraphId();
    const auto gName      = subGraphName(subGraphId);
    for (uint64_t o = 0; o < op_.nOutTensors(); ++o) {
      dtypeStrings[rowIndex]  = poprithms::ndarray::lcase(dtype({opId, o}));
      const auto devId        = op_.outDeviceId(o);
      deviceStrings[rowIndex] = device(devId).str();
      auto outRefs            = op_.derivedRefs(o);
      hasOutRefs |= !outRefs.empty();
      rootRefOf[rowIndex] = getStr(outRefs);
      ++rowIndex;
    }
    if (op_.nOutTensors() == 0) {
      ++rowIndex;
    }
  }

  cols.push_back({"Device", std::move(deviceStrings), colParams});

  // If there is no cross-graph referencing, do not append a column.
  if (hasOutRefs) {
    cols.push_back({"IsRootOf", std::move(rootRefOf), colParams});
  }
  cols.push_back({"Type", std::move(dtypeStrings), colParams});

  return cols;
}
DeviceId Graph::ipu(DeviceId id, uint64_t rank0, uint64_t rank1) {
  const auto &ipu_ = ipu(id);

  const auto subIntervals = ipu_.tiles().subIntervals(rank0, rank1);

  // We first check that this device doesn't already exist.
  for (uint64_t deviceId = 0; deviceId < devices.size(); ++deviceId) {
    if (device(deviceId).isIpu() && ipu(deviceId).tiles() == subIntervals) {
      return deviceId;
    }
  }

  DeviceId deviceId = devices.size();
  devices.emplace_back(std::make_unique<Ipu>(deviceId, subIntervals));
  return deviceId;
}

DeviceId Graph::createRemote(DeviceId ipu,
                             DType t,
                             const Shape &shape,
                             const RemoteOptions &ros) {
  DeviceId deviceId = devices.size();
  devices.emplace_back(
      std::make_unique<Remote>(deviceId, ipu, t, shape, ros));
  return deviceId;
}

std::vector<DeviceId> Graph::partition(DeviceId deviceId, uint64_t N) {

  const auto &dev = ipu(deviceId);

  if (dev.nTiles() % N != 0) {
    std::ostringstream oss;
    oss << "Cannot partition " << device(deviceId) << " into " << N
        << " parts, as it has " << dev.nTiles()
        << " which is not divisible by " << N << ". ";
    throw error(oss.str());
  }

  const auto perPart = dev.nTiles() / N;
  std::vector<DeviceId> parts;
  parts.reserve(N);
  for (uint64_t i = 0; i < N; ++i) {
    parts.push_back(ipu(deviceId, i * perPart, (i + 1) * (perPart)));
  }

  return parts;
}

const Ipu &Graph::ipu(DeviceId devId) const {
  const auto &dev = device(devId);
  const auto ipu  = dynamic_cast<const Ipu *>(&dev);
  if (!ipu) {
    std::ostringstream oss;
    oss << "Failed to cast " << dev << " to an Ipu. ";
    throw error(oss.str());
  }
  return *ipu;
}

DeviceIds Graph::ipuDevices() const {
  DeviceIds ipus;
  for (const auto &d : devices) {
    if (dynamic_cast<const Ipu *>(d.uptr.get())) {
      ipus.push_back(d.uptr->id());
    }
  }
  return ipus;
}

DeviceIds Graph::nonRootIpuDevices() const {
  DeviceIds allIpuDevices = ipuDevices();
  DeviceIds nonRootIpuDevices;
  nonRootIpuDevices.reserve(allIpuDevices.size());
  for (auto id : allIpuDevices) {
    if (id != rootIpu()) {
      nonRootIpuDevices.push_back(id);
    }
  }
  return nonRootIpuDevices;
}

DeviceIds Graph::remoteDevices() const {
  DeviceIds remotes;
  for (const auto &d : devices) {
    if (dynamic_cast<const Remote *>(d.uptr.get())) {
      remotes.push_back(d.uptr->id());
    }
  }
  return remotes;
}

const Remote &Graph::remote(DeviceId devId) const {
  const auto &dev   = device(devId);
  const auto remote = dynamic_cast<const Remote *>(&dev);
  if (!remote) {
    std::ostringstream oss;
    oss << "Failed to cast " << dev << " to a Remote. ";
    throw error(oss.str());
  }
  return *remote;
}

std::vector<std::pair<CallEvent, InIndex>>
Graph::indexedInCopies(const TensorId &tId) const {

  const auto &events = op(tId.opId()).inCopies(tId.outIndex());
  std::vector<std::pair<CallEvent, InIndex>> pairs;
  pairs.reserve(events.size());

  for (const auto &e : events) {
    const auto index = op(e.caller()).inIndex(CalleeTensorId(tId, e.index()));

    pairs.push_back({e, index});
  }
  return pairs;
}

std::vector<std::pair<CallEvent, OutIndex>>
Graph::indexedOutCopies(const TensorId &tId) const {

  std::vector<std::pair<CallEvent, OutIndex>> pairs;
  const auto &events = op(tId.opId()).outCopies(tId.outIndex());
  pairs.reserve(events.size());

  for (const auto &e : events) {
    const auto index =
        op(e.caller()).outIndex(CalleeTensorId(tId, e.index()));
    pairs.push_back({e, index});
  }

  return pairs;
}

TensorIds Graph::hostTensors() const {
  TensorIds hts;
  for (const auto opId : opIds()) {
    for (OutIndex o = 0; o < nOutTensors(opId); ++o) {
      if (isOnHost({opId, o})) {
        hts.push_back({opId, o});
      }
    }
  }
  return hts;
}

OpIds Graph::opsWithCallees() const {
  OpIds o;
  for (auto opId : opIds()) {
    if (op(opId).hasCallees()) {
      o.push_back(opId);
    }
  }
  return o;
}

std::vector<std::vector<uint64_t>> Graph::calleeGraph() const {
  std::vector<std::vector<uint64_t>> edges;
  const auto sources = opsWithCallees();
  edges.resize(nSubGraphs());
  for (auto src : sources) {
    for (auto dst : op(src).callees()) {
      auto &es = edges[subGraphId(src).get_u64()];
      if (std::find(es.begin(), es.end(), dst.get_u64()) == es.end()) {
        es.push_back(dst.get_u64());
      }
    }
  }
  return edges;
}

std::string Graph::str(const OpIds &opIds) const {
  std::vector<std::string> parts;
  for (auto opId : opIds) {
    parts.push_back(op(opId).str());
  }
  std::ostringstream oss;
  poprithms::util::append(oss, parts);
  return oss.str();
}

SubGraphIds Graph::reachable(const SubGraphIds &roots) const {

  auto calleeGraph_ = calleeGraph();

  auto toProcess = asUnsigned64s(roots);
  std::set<uint64_t> seen{toProcess.cbegin(), toProcess.cend()};

  std::vector<uint64_t> reachable = {};

  while (!toProcess.empty()) {
    auto nxt = toProcess.back();
    toProcess.pop_back();
    reachable.push_back(nxt);
    for (auto sgId : calleeGraph_.at(nxt)) {
      if (seen.count(sgId) == 0) {
        seen.insert(sgId);
        toProcess.push_back(sgId);
      }
    }
  }

  return asSubGraphIds(reachable);
}

std::map<SubGraphId, CallEvents> Graph::callEvents() const {
  std::map<SubGraphId, CallEvents> m;
  for (auto opId : opIds()) {
    auto callees_ = callees(opId);
    for (CalleeIndex i = 0; i < callees_.size(); ++i) {
      const auto sg = callees_[i.get()];
      CallEvent ce(opId, sg, i);
      auto found = m.find(sg);
      if (found == m.cend()) {
        m.insert({sg, {ce}});
      } else {
        found->second.push_back(ce);
      }
    }
  }
  return m;
}

bool Graph::isRunnable(SubGraphId sgId) const {
  return std::find(runnable_.cbegin(), runnable_.cend(), sgId) !=
         runnable_.cend();
}

void Graph::setRunnable(const SubGraphIds &rids) {

  auto sorted = poprithms::util::unisorted(rids);
  if (sorted == runnable_) {
    return;
  }

  if (!runnable_.empty()) {
    std::ostringstream oss;
    oss << "This Graph already has a set of runnable sub-graphs, "
        << runnable_ << ". This set should be only be set once "
        << "(the method setRunnable is not 'incremental'). "
        << "Bailing on call to set runnables to " << rids << '.';
    throw error(oss.str());
  }

  runnable_ = poprithms::util::unisorted(rids);
}

CallEvent Graph::callEvent(OpId opId) const {
  auto callees_ = op(opId).callees();
  if (callees_.size() != 1) {
    std::ostringstream oss;
    oss << "Invalid call to method callEvent(OpId=" << opId << ") where "
        << opId << " is the id of the op " << op(opId)
        << ". This method can only be used for ops with 1 callee. ";
    throw error(oss.str());
  }
  return CallEvent(opId, callees_[0], CalleeIndex(0));
}

const Device &Graph::device(DeviceId id) const {
  return *devices.at(id.get_u64()).uptr;
}

uint64_t Graph::nbytes(const TensorId &id) const {
  return nelms_u64(id) * poprithms::ndarray::nbytes(dtype(id));
}

void Graph::setInitialValue(const TensorId &tId,
                            uint64_t replica,
                            const poprithms::compute::host::Tensor &v) {
  op(tId.opId()).setInitialValue(replica, tId.outIndex(), v);
}

DTypes Graph::dtypes(const TensorIds &tIds) const {
  DTypes ts;
  ts.reserve(tIds.size());
  for (const auto &tId : tIds) {
    ts.push_back(dtype(tId));
  }
  return ts;
}

bool Graph::gradientPropagates(const OpTraversal &ot) const {

  if (op(ot.opId()).inIsFixedPoint(ot.inIndex()) ||
      op(ot.opId()).outIsFixedPoint(ot.outIndex())) {
    return false;
  }

  auto props = op(ot.opId()).gradientPropagates(ot.outIndex(), ot.inIndex());

  return props;
}

bool Graph::gradientPropagates(const TensorId &id) const {
  for (uint64_t i = 0; i < nInTensors(id.opId()); ++i) {
    if (gradientPropagates(
            OpTraversal{InIndex(i), id.opId(), id.outIndex()})) {
      return true;
    }
  }
  return false;
}

Shape Graph::getHostShape(CircularBufferCount ff,
                          ReplicationFactor rf,
                          const Shape &ipuShape) {

  auto &&s1 = ipuShape.get();

  std::vector<int64_t> s0;
  s0.reserve(2 + s1.size());
  s0.push_back(ff.get());
  s0.push_back(rf.get_i64());
  s0.insert(s0.end(), s1.cbegin(), s1.cend());
  return Shape(std::move(s0));
}

DeviceId Graph::deviceId(SubGraphId sgId) const {

  // All sub-graphs reachable via ops with callees:
  auto sgIds = reachable({sgId});

  TensorIds all;
  for (auto sgId_ : sgIds) {
    auto nxt = tensorIds(sgId_);
    all.insert(all.end(), nxt.cbegin(), nxt.cend());
  }

  return deviceIdByUnanimity(all);
}

bool Graph::multiOutTypeSpecificEqualTo(const multiout::Graph &rhs) const {
  const auto r                  = dynamic_cast<const Graph *>(&rhs);
  const auto atComputeLevel     = computeTypeSpecificEqualTo(*r);
  const auto atSchedulabelLevel = schedulableTypeSpecificEqualTo(*r);
  return atSchedulabelLevel && atComputeLevel;
}

bool Graph::isConstInit(OpId opId) const {
  return dynamic_cast<const ConstInit *>(&op(opId));
}

bool Graph::isVarInit(OpId opId) const {
  return dynamic_cast<const VarInit *>(&op(opId));
}

HostTensor Graph::constInitValue(OpId opId) const {
  return castOrThrow<ConstInit>(opId)->value();
}

void Graph::verifyComputeDerivedGraphValid() const {
  for (const auto opId : opIds()) {
    op(opId).computeDerivedVerifyValid();
  }
}

TensorId Graph::refFrom_(const TensorId &tId, SubGraphId destination) {
  return tRefFrom<RefFrom>(tId, destination);
}

void Graph::verifyComputeDerivedOpValid(OpId opId) const {
  op(opId).computeDerivedVerifyValid();
}

void Graph::setUserManagedHost(const TensorId &tId, bool b) {

  auto asVarInit = dynamicMutableCast<VarInit>(tId.opId());
  if (!asVarInit) {
    std::ostringstream oss;
    oss << "The creator of the op " << tId << " is " << op(tId.opId())
        << ", which is not a VarInit. "
        << "Only VarInit ops can have the flag for 'user managed host' set. ";
    throw error(oss.str());
  }

  asVarInit->setUserManagedHost(b);
}

// TODO(T63457) add the tests which currently depend on ops which have not yet
// been added.
void Graph::multiOutTypeSpecificRemoveOutputs(
    OpId opId,
    const ContiguousOutIndexSubset &coin,
    const OptionalTensorIds &subs) {

  // if output is being removed, return.
  if (coin.nSubset() == 0) {
    return;
  }

  {
    // The op which is having some outputs removed:
    const auto &op_ = op(opId);

    // For all of the outputs, if the output is a root reference of tensor(s)
    // in different sub-graphs, adjust the stored root reference of the
    // derived tensors. This involves a shifting down of output indices to
    // fill in the gaps.
    //
    // Example:
    //   suppose this op has 3 outputs and output #0 is being removed, and
    //   output #1 has a derived reference tId. Then the stored root reference
    //   of tId will change from (opId, 1) to (opId, 0).
    //
    // If an output with a derived reference is removed, an error will be
    // thrown. The derived reference needs to adjusted before this call to
    // remove the root reference.
    for (OutIndex o = 0; o < op_.nOutTensors(); ++o) {
      for (auto dr : op_.derivedRefs(o)) {
        op(dr.opId()).resetRootRef(dr.outIndex(), {opId, coin.inSubset(o)});
      }
    }

    // Suppose the op #opId, which is having outputs removed, has callees.
    //
    // Suppose that output #o of the op is copied out of callee #ci. Then
    // there is some op in the callee #ci which creates the tensor being
    // copied out at index #o. This creator (#toAmend) stores the call event
    // with this out-copy as an attribute (see the Op class), and this call
    // event should be removed now that #o is no longer copied out.
    for (auto o : coin.toRemove()) {
      for (CalleeIndex ci = 0; ci < op_.nCallees(); ++ci) {
        if (op_.isCopiedOut(o, ci)) {
          auto toAmend = op_.srcInCallee(o, ci);
          auto ce      = CallEvent(opId, op_.callee(ci), ci);
          op(toAmend.opId()).removeOutCopy(toAmend.outIndex(), ce);
        }
      }
    }
  }

  auto getSub = [&subs](OutIndex o) { return subs.at(o.get()).value(); };

  // Suppose the op #opId, which is having outputs removed, has an output #o
  // which is copied into during a call event. So #opId is probably a
  // VarInit in a callee sub-graph, although this cannot be assumed.
  //
  // Here we transfer the destination of in-copy from (opId, #o) to the
  // substitute provided to this method.
  for (auto o : coin.toRemove()) {
    for (auto ce : op(opId).inCopies(o)) {
      auto removedId = CalleeTensorId({opId, o}, ce.index());
      auto sub       = getSub(o);
      auto &caller   = op(ce.caller());
      op(sub.opId()).insertInCopy(sub.outIndex(), ce);
      caller.resetCalleeTensorId(caller.inIndex(removedId),
                                 {sub, ce.index()});
      op(opId).removeInCopy(o, ce);
    }
  }

  // Suppose the #opId which is having outputs removed has an output #o
  // which is copied out from during a call event. So #opId is probably a
  // final op in a callee sub-graph, although we don't assume this.
  //
  // Here we transfer the source of the out-copy from (opId, #o) to the
  // substitute provided to this method.
  for (auto o : coin.toRemove()) {
    for (auto ce : op(opId).outCopies(o)) {
      auto removedId = CalleeTensorId({opId, o}, ce.index());
      auto sub       = getSub(o);
      auto &caller   = op(ce.caller());
      op(sub.opId()).insertOutCopy(sub.outIndex(), ce);
      caller.resetOutSource(caller.outIndex(removedId), ce.index(), sub);
      op(opId).removeOutCopy(o, ce);
    }
  }

  // Shift indices downwards to fill in the gaps.
  for (OutIndex o = 0; o < nOutTensors(opId); ++o) {
    for (auto ce : op(opId).inCopies(o)) {
      CalleeTensorId toRemove({opId, o}, ce.index());
      CalleeTensorId shiftedId({opId, coin.inSubset(o)}, ce.index());
      auto &caller = op(ce.caller());
      caller.resetCalleeTensorId(caller.inIndex(toRemove), shiftedId);
    }
    for (auto ce : op(opId).outCopies(o)) {
      auto &caller = op(ce.caller());
      caller.resetOutSource(caller.outIndex({{opId, o}, ce.index()}),
                            ce.index(),
                            {opId, coin.inSubset(o)});
    }
  }

  // Op specific work.
  op(opId).computeOpRemoveOutputs(coin);
}

void Graph::multiOutTypeSpecificRemoveInputs(
    OpId opId,
    const ContiguousInIndexSubset &coin) {

  const auto &op_ = op(opId);

  // If the input #i to the op #opId is being removed, and if the input #i is
  // copied to a callee sub-graph (#opId has callees), then signal to the
  // tensor in the callee that is copied to that it is no longer being copied
  // to.
  for (InIndex i : coin.toRemove()) {

    if (op_.isCopyToCalleeInIndex(i)) {

      // Tensor being copied to in callee:
      auto toAmend     = op_.dstInCallee(i.get());
      auto tId         = toAmend.tId();
      auto calleeIndex = toAmend.calleeIndex();
      op(tId.opId())
          .removeInCopy(
              tId.outIndex(),
              CallEvent(opId, op_.callee(calleeIndex), calleeIndex));
    }
  }

  // perform any op specific removal work.
  op(opId).computeOpRemoveInputs(coin);
}

// This is the final override of the virtual method declared in the base
// class, common::schedulable::Graph::insertBinBoundary.
//
// As per the definition in schedulable:Graph, this virtual method must:
//   ' ...Insert a "null" Op which serves no purpose other than to separate
//   bins of Ops"... '
//
// The Nop fits this requirement, so we use a Nop as a boundary to separate
// groups of ops when scheduling.
OpId Graph::insertBinBoundary(SubGraphId sgId) {
  return createComputeOp<Nop>({}, sgId, {});
}

bool Graph::modifies(OpId opId, InIndex inIndex) const {
  return op(opId).modifies(inIndex);
}

bool Graph::aliases(OpId opId, InIndex inIndex, OutIndex outIndex) const {
  return op(opId).aliases(inIndex, outIndex);
}

std::map<OpId, OpIds>
Graph::schedulableDerivedSpecificConstraints(const OpIds &opIds) const {
  return poprithms::common::compute::AliasGraphQuerier::
      makeModifiersFinalConsumers(*this, opIds);
}

} // namespace compute
} // namespace common
} // namespace poprithms
