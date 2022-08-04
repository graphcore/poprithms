// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/pipeline.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>
#include <poprithms/program/pipeline/imutator.hpp>
#include <poprithms/program/pipeline/iquerier.hpp>
#include <poprithms/program/pipeline/objective.hpp>
#include <poprithms/program/pipeline/pipeline.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace {

/**
 * Interface implementation for the compute::Graph class.
 * */
class Querier : public poprithms::program::pipeline::IQuerier {

public:
  Querier(const Graph &g_, SubGraphId sgId_) : g(g_), sgId(sgId_) {}

  uint64_t nOutTensors(OpId opId) const final { return g.nOutTensors(opId); }

  ConsumptionIds consumptionIds(const TensorId &tId) const final {
    return g.consumptionIds(tId);
  }

  OpIds schedule() const final { return g.vanillaSubGraphSchedule(sgId); }

  TensorIds inTensorIds(OpId opId) const final { return g.inTensorIds(opId); }

  Shape shape(const TensorId &tId) const { return g.shape(tId); }

private:
  const Graph &g;
  SubGraphId sgId;
};

/**
 * Interface implementation for the compute::Graph class.
 * */
class Mutator : public poprithms::program::pipeline::IMutator {

public:
  Mutator(SlickGraph &g_, const AcclTypedObjective &o_) : g(g_), o(o_) {}

  OpId call(SubGraphId caller, SubGraphId callee) const final {
    return SubGraph(caller, g).call(callee, {}, {});
  }

  OpId repeat(SubGraphId caller,
              SubGraphId callee,
              uint64_t tripCount) const final {
    return SubGraph(caller, g).repeat(callee, tripCount, {}, {}, {});
  }

  SubGraphId createSubGraph(const std::string &x) const final {
    return g.createSubGraphId(x);
  }

  SubGraphId createInOrderSubGraph(const std::string &x) const final {
    auto sg0 = g.createSubGraphId(x);
    g.toggleEager(sg0, true);
    return sg0;
  }

  OpId clone(OpId opId,
             const TensorIds &ins,
             SubGraphId sg,
             const DeviceIds &outDevIds) const final {
    return g.clone(opId, ins, sg, outDevIds);
  }

  TensorId refFrom_(const TensorId &tId, SubGraphId sg) const final {
    return g.refFrom_(tId, sg);
  }

  TensorId copy(const TensorId &tId, DeviceId devId) const final {
    return Tensor(tId, &g).copy(devId);
  }

  TensorId copy_(const TensorId &src, const TensorId &dst) const final {
    return Tensor(dst, &g).copyFrom_({src, &g});
  }

  TensorId variable(DType dt,
                    const Shape &s,
                    DeviceId devId,
                    SubGraphId sgId) const final {
    return SubGraph(sgId, g).variable(dt, s, devId);
  }

  TensorId variableLike(const TensorId &t0, const Shape &s) const final {
    return Tensor(t0, &g).variable(s);
  }

  TensorId variableLike(const TensorId &t0,
                        DeviceId dId,
                        SubGraphId sgId) const final {
    return Tensor(t0, &g).variable(dId, sgId);
  }

  TensorId dynamicAt(const TensorId &t0, const TensorId &index) const final {
    return Tensor(t0, &g).dynamicAt(Tensor(index, &g));
  }

  TensorId updateAt_(const TensorId &sliceable,
                     const TensorId &slice,
                     const TensorId &index) const final {
    Tensor tSliceable(sliceable, &g);
    Tensor tSlice(slice, &g);
    Tensor tIndex(index, &g);
    return tSliceable.updateAt_(tSlice, tIndex);
  }

  TensorId add(const TensorId &tId, uint64_t v) const final {
    auto t = Tensor(tId, &g);
    return t.add(t.constant(v));
  }

  TensorId sub(const TensorId &tId, uint64_t v) const final {
    auto t = Tensor(tId, &g);
    return t.sub(t.constant(v));
  }

  TensorId add_(const TensorId &tId, uint64_t v) const final {
    auto t = Tensor(tId, &g);
    return t.add_(t.constant(v));
  }

  TensorId zero_(const TensorId &tId) const final {
    return Tensor(tId, &g).zero_();
  }

  TensorId modulo(const TensorId &tId, uint64_t v) const final {
    return Tensor(tId, &g).modulo(v);
  }

  TensorId initAccumulator_(const TensorId &unpipelined,
                            const TensorId &tId) const final {
    switch (o.acclType(unpipelined)) {
    case PipelineAcclType::Sum: {
      return Tensor(tId, &g).zero_();
    }
    case PipelineAcclType::RunningMean: {
      return Tensor(tId, &g).zero_();
    }

    case PipelineAcclType::Max: {
      return Tensor(tId, &g).setToLowest_();
    }
    }

    throw error("Unhandlded Accumulation Type");
  }

  TensorId accumulate(const TensorId &unpipelined,
                      const TensorId &partial,
                      const TensorId &toUpdate,
                      const TensorId &accumulationCount) const final {

    switch (o.acclType(unpipelined)) {
    case PipelineAcclType::Sum: {
      return Tensor(toUpdate, &g).add_(Tensor(partial, &g));
    }
    case PipelineAcclType::Max: {
      return Tensor(toUpdate, &g).max_(Tensor(partial, &g));
    }
    case PipelineAcclType::RunningMean: {
      auto tToUpdate   = Tensor(toUpdate, &g);
      auto i           = Tensor(accumulationCount, &g).to(tToUpdate.dtype());
      auto iPlusOneInv = i.add(1).inv();
      Tensor tPartial(partial, &g);
      return tToUpdate.mul_(i * iPlusOneInv).add_(tPartial.mul(iPlusOneInv));
    }
    }

    throw error("Unhandlded Accumulation Type");
  }

  void setName(OpId opId, const std::string &n) const final {
    g.setName(opId, n);
  }

  std::string name(OpId opId) const final { return g.getName(opId); }

private:
  SlickGraph &g;
  const AcclTypedObjective &o;
};

} // namespace

PipelineAcclType AcclTypedObjective::acclType(const TensorId &tId) const {
  auto found = acclTypes_.find(tId);
  if (found == acclTypes_.cend()) {
    std::ostringstream oss;
    oss << "There is no PipelineAcclType registered for the (unpipelined) "
           "tensor "
        << tId << ".";
    throw error(oss.str());
  }
  return found->second;
}

AcclTypedObjective::AcclTypedObjective(
    const std::map<OpId, PipelineStage> &stages,
    const DeviceIds &stageDevices,
    uint64_t nToAccumulate,
    const TensorIds &toAccumulate,
    const PipelineAcclTypes &acclTypes,
    const TensorIds &streamingInputs)
    : Objective(stages,
                stageDevices,
                nToAccumulate,
                toAccumulate,
                streamingInputs) {
  if (acclTypes.size() != toAccumulate.size()) {
    std::ostringstream oss;
    oss << "Incompatible numbers of types (" << acclTypes.size()
        << ") and accumulate tensor ids (" << toAccumulate.size()
        << "). They should be the same.";
    throw error(oss.str());
  }
  for (uint64_t i = 0; i < acclTypes.size(); ++i) {
    acclTypes_.insert({toAccumulate.at(i), acclTypes.at(i)});
  }
}

Pipeline::Pipeline(SlickGraph &g,
                   SubGraphId sgId,
                   const AcclTypedObjective &obj)
    : poprithms::program::pipeline::Pipeline(obj,
                                             Querier(g, sgId),
                                             Mutator(g, obj)) {}

} // namespace compute
} // namespace common
} // namespace poprithms
