// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>
#include <string>

#include <program/pipeline/error.hpp>

#include <poprithms/program/pipeline/guide.hpp>
#include <poprithms/program/pipeline/imutator.hpp>
#include <poprithms/program/pipeline/iquerier.hpp>
#include <poprithms/program/pipeline/objective.hpp>
#include <poprithms/program/pipeline/pipeline.hpp>

namespace poprithms {
namespace program {
namespace pipeline {

void IMutator::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

void IQuerier::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

Objective::Objective(const std::map<OpId, PipelineStage> &stages,
                     const DeviceIds &stageDevices,
                     const uint64_t nToAccumulate,
                     const TensorIds &toAccumulate,
                     const TensorIds &streamingInputs)
    : stages_(stages), stageDevices_(stageDevices),
      toAccumulate_(toAccumulate), streamingInputs_(streamingInputs),
      nToAccumulate_(nToAccumulate) {

  if (nToAccumulate < nStages()) {
    std::ostringstream oss;
    oss << "Stream size (nToAccumulate=" << nToAccumulate
        << " cannot be smaller than the number of stages (" << nStages()
        << ").";
    throw error(oss.str());
  }

  for (auto &&[k, w] : stages) {
    if (static_cast<uint64_t>(w.get()) >= stageDevices.size()) {
      std::ostringstream oss;
      oss << "The op " << k << " is in pipeline stage " << w << ". "
          << "There are devices provided for only " << stageDevices.size()
          << " stages, however. Failed to construct Objective.";
      throw error(oss.str());
    }
  }

  mustAccumulate_ = {toAccumulate_.cbegin(), toAccumulate_.cend()};
}

DeviceId Objective::deviceId(PipelineStage ps) const {
  if (static_cast<uint64_t>(ps.get()) >= stageDevices_.size()) {
    std::ostringstream oss;
    oss << "The pipeline stage " << ps
        << " does not have a device registered. "
        << "There are devices for stages in the interval [0,"
        << stageDevices_.size() << ").";
  }
  return stageDevices_.at(ps.get());
}

PipelineStage Objective::stage(OpId opId) const {
  auto found = stages_.find(opId);
  if (found == stages_.cend()) {
    std::ostringstream oss;
    oss << "There is no pipeline stage registered for the op " << opId << '.';
    throw error(oss.str());
  }
  return found->second;
}

TensorIds Objective::mustAccumulate(PipelineStage ps) const {
  TensorIds tIds;
  for (auto &&x : toAccumulate_) {
    if (stage(x) == ps) {
      tIds.push_back(x);
    }
  }
  return tIds;
}

TensorIds IQuerier::outTensorIds(OpId opId) const {
  TensorIds outIds;
  const auto n = nOutTensors(opId);
  outIds.reserve(n);
  for (OutIndex o = 0; o < n; ++o) {
    outIds.push_back({opId, o});
  }
  return outIds;
}

OutIndices IQuerier::outIndices(OpId opId) const {
  OutIndices ois;
  const auto n = nOutTensors(opId);
  ois.reserve(n);
  for (OutIndex o = 0; o < n; ++o) {
    ois.push_back(o);
  }
  return ois;
}

PipelineStages Objective::ascendingStages() const {
  PipelineStages ss;
  ss.reserve(nStages());
  for (uint64_t i = 0; i < nStages(); ++i) {
    ss.push_back({i});
  }
  return ss;
}

Guide::Guide(const IQuerier &gq, const Objective &obj)
    : querier(gq), objective(obj) {

  auto fullSched = gq.schedule();
  stageSchedules_.resize(objective.nStages());
  for (auto &&op : fullSched) {
    stageSchedules_[obj.stage(op).get()].push_back(op);
  }

  for (auto &&op : fullSched) {
    auto stage0 = obj.stage(op);
    for (auto o : gq.outTensorIds(op)) {
      for (auto c : gq.consumptionIds(o)) {
        auto s0 = obj.stage(c.opId());
        if (s0.get() < stage0.get()) {
          std::ostringstream oss;
          oss << "The op " << op << " is in pipeline stage " << stage0
              << ". It has a consumer, " << c.opId()
              << " (consumption id = " << c << ") which is in pipeline stage "
              << s0 << ". But " << s0 << " < " << stage0
              << ". Consumers cannot be in later stages than producers.";
          throw error(oss.str());
        }
      }
    }
  }

  const auto vFromStreamIn =
      poprithms::common::multiout::depthFirstForwardTensors(
          querier, obj.streamingInputs(), [](auto &&) { return true; });

  fromStreamIn = {vFromStreamIn.cbegin(), vFromStreamIn.cend()};
}

bool Guide::isUnchanging(const TensorId &tId) const {
  return fromStreamIn.count(tId) == 0;
}

PipelineStages Guide::consumers(const TensorId &tId) const {
  std::set<PipelineStage> ps;
  for (auto c : querier.consumptionIds(tId)) {
    if (objective.stage(c.opId()) != objective.stage(tId)) {
      ps.insert(objective.stage(c.opId()));
    }
  }
  return PipelineStages(ps.cbegin(), ps.cend());
}

bool Guide::isUnchanging(OpId opId) const {
  for (OutIndex o = 0; o < querier.nOutTensors(opId); ++o) {
    if (!isUnchanging({opId, o})) {
      return false;
    }
  }
  for (auto i : querier.inTensorIds(opId)) {
    if (!isUnchanging(i)) {
      return false;
    }
  }
  return true;
}

OpIds Guide::unchangingScheduled(PipelineStage ps) const {
  OpIds opIds;
  for (auto &&x : scheduled(ps)) {
    if (isUnchanging(x)) {
      opIds.push_back(x);
    }
  }
  return opIds;
}

OpIds Guide::changingScheduled(PipelineStage ps) const {
  OpIds opIds;
  for (auto &&x : scheduled(ps)) {
    if (!isUnchanging(x)) {
      opIds.push_back(x);
    }
  }
  return opIds;
}

TensorIds Guide::tensorIds(PipelineStage ps) const {
  TensorIds tIds;
  for (auto op : stageSchedules_.at(ps.get())) {
    for (OutIndex o = 0; o < querier.nOutTensors(op); ++o) {
      tIds.push_back({op, o});
    }
  }
  return tIds;
}

TensorId Pipeline::getInStage(PipelineStage ps,
                              const TensorId &unpipelined) const {
  if (guide.isUnchanging(unpipelined)) {
    for (auto &&x : unchangingTensorsInPipelineStages.at(unpipelined)) {
      if (x.first == ps) {
        return x.second;
      }
    }
    std::ostringstream oss;
    oss << "The (unchanging) unpipelined tensor " << unpipelined
        << " was not found in the pipeline stage " << ps << '.';
    throw error(oss.str());
  }

  if (objective.stage(unpipelined) == ps) {
    return {stageClones.at(unpipelined.opId()), unpipelined.outIndex()};
  }

  auto &&cs = interStageTensorMappings.at(unpipelined);
  for (auto &&x : cs) {
    if (x.to == ps) {
      return x.destination;
    }
  }
  throw error("should be unreachable (copy dst not found)");
}

TensorIds Pipeline::getInStage(PipelineStage ps,
                               const TensorIds &tIds) const {
  TensorIds ids;
  for (auto &&tId : tIds) {
    ids.push_back(getInStage(ps, tId));
  }
  return ids;
}

TensorIds Pipeline::getUnchangingTensors(const TensorIds &unpipelined,
                                         DeviceId dId) const {
  TensorIds ids;
  for (auto x : unpipelined) {
    auto nxt = [&]() {
      for (auto &&y : unchangingTensorsInInit.at(x)) {
        if (y.first == dId) {
          return y.second;
        }
      }
      std::ostringstream oss;
      oss << "There is no unchanging tensor on device " << dId
          << " for the unpipelined tensor " << unpipelined << ".";
      throw error(oss.str());
    }();
    ids.push_back(nxt);
  }
  return ids;
}

void Pipeline::setUnchangingTensors(const IMutator &mutator,
                                    const IQuerier &querier) {
  for (auto stage : objective.ascendingStages()) {
    auto stageDev = objective.deviceId(stage);
    auto stageSg  = stageSubGraphs.at(stage.get());
    for (auto opId : guide.unchangingScheduled(stage)) {
      auto initIns =
          getUnchangingTensors(querier.inTensorIds(opId), stageDev);
      auto initOp =
          mutator.clone(opId,
                        initIns,
                        sgInitialize,
                        DeviceIds(querier.nOutTensors(opId), stageDev));

      mutator.setName(initOp,
                      "clone of (opId=" + std::to_string(opId.get()) + ")");

      for (auto o : querier.outIndices(opId)) {

        TensorId unpipelinedId{opId, o};
        TensorId tInInit({initOp, o});

        unchangingTensorsInInit.insert(
            {unpipelinedId, {{stageDev, {initOp, o}}}});

        unchangingTensorsInPipelineStages.insert(
            {unpipelinedId, {{stage, mutator.refFrom_(tInInit, stageSg)}}});

        for (auto &&c : querier.consumptionIds(unpipelinedId)) {
          auto cStage   = objective.stage(c.opId());
          auto cDev     = objective.deviceId(cStage);
          auto cStageSg = stageSubGraphs.at(cStage.get());
          if (cStage != stage) {

            TensorId tInitOnDevice = [&]() -> TensorId {
              auto &s = unchangingTensorsInInit.at(unpipelinedId);
              for (auto x : s) {
                if (x.first == cDev) {
                  return x.second;
                }
              }
              auto b = mutator.copy(tInInit, cDev);
              s.push_back({cDev, b});
              return b;
            }();

            [&]() {
              auto &s = unchangingTensorsInPipelineStages.at(unpipelinedId);
              for (auto x : s) {
                if (x.first == cStage) {
                  return x.second;
                }
              }
              auto b = mutator.refFrom_(tInitOnDevice, cStageSg);
              s.push_back({cStage, b});
              return b;
            }();
          }
        }
      }
    }
  }
}

void Pipeline::setChangingTensors(const IMutator &mutator,
                                  const IQuerier &querier) {
  for (auto stage : objective.ascendingStages()) {

    const auto devStage      = objective.deviceId(stage);
    const auto stageSubGraph = stageSubGraphs[stage.get()];

    // Construct all the stage sub-graphs ops.
    for (auto opId : guide.changingScheduled(stage)) {
      auto ins = getInStage(stage, querier.inTensorIds(opId));
      auto cl  = mutator.clone(
          opId,
          getInStage(stage, querier.inTensorIds(opId)),
          stageSubGraph,
          std::vector<DeviceId>(querier.nOutTensors(opId), devStage));
      mutator.setName(cl,
                      "clone of (opId=" + std::to_string(opId.get()) + ")");
      stageClones.insert({opId, cl});
    }

    // Manage the stash and restore operations.
    for (const auto &tId : guide.tensorIds(stage)) {

      // Unchanging ops are never stashed as they always have the same value.
      if (guide.isUnchanging(tId)) {
        continue;
      }

      interStageTensorMappings.insert({tId, {}});
      auto t0 = getInStage(stage, tId);

      // All the stages which have an op which consumes #tId (other than the
      // stage of #tId itself).
      auto consumers_ = guide.consumers(tId);

      // If t0 is consumed in a pipeline stage other than its own.
      if (!consumers_.empty()) {

        // The number of pipeline stages before the final use of #tId.
        auto maxDorm = consumers_.back().get() - stage.get();

        // The stash size required.
        auto stashSize = maxDorm - 1;

        TensorId stash{-1, -1};
        TensorId stashInRestore{-1, -1};
        if (stashSize > 0) {
          stash =
              mutator.variableLike(t0, querier.shape(t0).prepend(stashSize));
          mutator.setName(stash.opId(), "stash of " + tId.str());
        }
        if (stashSize > 0) {
          stashInRestore = mutator.refFrom_(stash, sgRestoreStashTick);
        }

        auto indexInRestore = cycleIndices.at(devStage);

        for (const auto &consumer : consumers_) {

          // The device of the consuming stage.
          auto devConsumer = objective.deviceId(consumer);

          // The restored tensor.
          auto restore = [&] {
            const auto dist       = consumer.get() - stage.get();
            const auto readOffset = stashSize + 1 - dist;

            // If the consumer is in the next pipeline stage, there is no
            // stash/restore required.
            if (dist == 1) {
              return t0;
            }
            auto sliceIndex = indexInRestore;
            sliceIndex      = mutator.add(sliceIndex, readOffset);
            sliceIndex      = mutator.modulo(sliceIndex, stashSize);
            mutator.setName(sliceIndex.opId(), "restore index");
            auto slice = mutator.dynamicAt(stashInRestore, sliceIndex);
            mutator.setName(slice.opId(), "restore");
            return slice;
          }();

          if (devConsumer != devStage) {
            auto restoreCopyDst = mutator.variableLike(
                restore, devConsumer, stageSubGraphs[consumer.get()]);
            mutator.setName(restoreCopyDst.opId(), "restore destination");
            interStageTensorMappings[tId].push_back(
                {consumer, restore, restoreCopyDst});
            auto dst_ = mutator.refFrom_(restoreCopyDst, sgCopy);
            auto src_ = mutator.refFrom_(restore, sgCopy);
            auto cp   = mutator.copy_(src_, dst_);

            std::ostringstream oss;
            oss << "dev(" << devStage << "->" << devConsumer << ") stage("
                << stage << "->" << consumer << ")";

            mutator.setName(cp.opId(), oss.str());
          }

          // If the consuming stage's device is the same as the producing
          // stage's device, there is no inter-device copy required.
          else {
            interStageTensorMappings[tId].push_back(
                {consumer,
                 restore,
                 mutator.refFrom_(restore, stageSubGraphs[consumer.get()])});
          }
        }

        // Stash operation.
        if (stashSize > 0) {
          mutator.updateAt_(stashInRestore,
                            mutator.refFrom_(t0, sgRestoreStashTick),
                            mutator.modulo(indexInRestore, stashSize));
        }
      }
    }
  }
}

void Pipeline::setAccumulators(const IMutator &mutator, const IQuerier &) {

  for (auto stage : objective.ascendingStages()) {
    auto stageSubGraph = stageSubGraphs[stage.get()];

    for (const auto &tId : objective.mustAccumulate(stage)) {
      auto t0 = getInStage(stage, tId);

      auto accl =
          mutator.variableLike(t0, objective.deviceId(stage), stageSubGraph);
      mutator.setName(accl.opId(), "accumulator");

      auto acclCount = mutator.sub(indexInStage(mutator, stage), stage.get());
      auto accumulator =
          mutator.accumulate(tId, getInStage(stage, tId), accl, acclCount);
      mutator.setName(accumulator.opId(), "accumulate");
      accumulators.insert({tId, accl});
    }
  }
}

Pipeline::Pipeline(const Objective &objective_,
                   const IQuerier &q,
                   const IMutator &m)

    : objective(objective_), guide(q, objective_),

      sgCopy(m.createSubGraph("sg-copy")),
      sgMain(m.createInOrderSubGraph("sg-main")),
      sgRestoreStashTick(m.createSubGraph("sg-stash-restore-tick")),
      sgInitialize(m.createSubGraph("sg-initialize")) {

  // create sub-graphs of various sorts.
  for (auto stage : objective.ascendingStages()) {
    auto devId = objective.deviceId(stage);

    auto stageSubGraph =
        m.createSubGraph("sg-stage-" + std::to_string(stage.get()));

    stageSubGraphs.push_back(stageSubGraph);

    auto found = cycleIndices.find(devId);
    if (found == cycleIndices.cend()) {
      cycleIndices.insert(
          {devId,
           m.variable(DType::Unsigned32, {}, devId, sgRestoreStashTick)});
    }
  }

  setUnchangingTensors(m, q);

  setChangingTensors(m, q);

  setAccumulators(m, q);

  for (auto &[k, v] : accumulators) {
    auto acclInit = m.initAccumulator_(k, m.refFrom_(v, sgInitialize));
    m.setName(acclInit.opId(), "init accumulator");
  }

  for (auto &v : cycleIndices) {
    auto indexInit = m.zero_(m.refFrom_(v.second, sgInitialize));
    m.setName(indexInit.opId(), "cycle index <- 0");
  }

  for (auto &v : cycleIndices) {
    auto incrIndex = m.add_(v.second, 1);
    m.setName(incrIndex.opId(), "++(cycle index)");
  }

  // ramp up. repeat. ramp down.

  m.call(sgMain, sgInitialize, "initialization");

  // ramp up:
  for (uint64_t i = 1; i < objective.nStages(); ++i) {
    for (uint64_t j = i; j > 0; --j) {
      auto stage = j - 1;
      m.call(sgMain,
             stageSubGraphs.at(stage),
             "ramp up stage " + std::to_string(stage));
    }

    m.call(sgMain, sgRestoreStashTick, "restore, stash, increment");

    m.call(sgMain, sgCopy, "inter device copies");
  }

  // repeat:
  auto sgRep = m.createInOrderSubGraph("sg-full-pipe");
  for (uint64_t j = objective.nStages(); j > 0; --j) {
    m.call(sgRep,
           stageSubGraphs.at(j - 1),
           "full pipeline stage " + std::to_string(j - 1));
  }
  m.call(sgRep, sgRestoreStashTick, "restore, stash, increment");
  m.call(sgRep, sgCopy, "inter-ipu copies");

  uint64_t rptCnt = objective.nToAccumulate() - objective.nStages() + 1;
  auto rOp        = m.repeat(sgMain, sgRep, rptCnt);
  m.setName(rOp, "full pipeline repeat");

  // ramp down:
  for (uint64_t i = 1; i < objective.nStages(); ++i) {
    for (uint64_t j = objective.nStages() - 1; j >= i; --j) {
      m.call(sgMain,
             stageSubGraphs.at(j),
             "ramp down stage " + std::to_string(j));
    }
    m.call(sgMain, sgRestoreStashTick, "restore, stash, increment");
    m.call(sgMain, sgCopy, "inter-ipu copies");
  }
}

TensorId Pipeline::accumulatorInStage(const TensorId &unpipelined) const {
  auto found = accumulators.find(unpipelined);
  if (found == accumulators.cend()) {
    std::ostringstream oss;
    oss << "The tensor " << unpipelined
        << " does not have an accumulation tensor stored for it.";
    throw error(oss.str());
  }
  return found->second;
}

} // namespace pipeline
} // namespace program
} // namespace poprithms
