// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_REQUIREDIDS_HPP
#define POPRITHMS_COMMON_COMPUTE_REQUIREDIDS_HPP

#include <set>

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/calleeindex.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

using poprithms::autodiff::core::GradInfo;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::TensorId;
using poprithms::common::schedulable::SubGraphId;
using poprithms::program::callstack::CalleeIndex;

/**
 * The ids of the tensors required to perform autodiff on a graph.
 *
 * This class 'wraps' a reference to a std::set<TensorId> which is used in
 * poprithms::autodiff::guide::GraphInfo::extendAutodiffRequiredTensors.
 *
 * It is not possible to use the 'raw' std::set<TensorId> directly at this
 * level of the autodiff project as certain ops need additional information
 * (GradInfos) to determine which tensors they require. Hence this class also
 * contains a reference to a GradInfos object.
 * */
class RequiredIds {
private:
  std::set<TensorId> &tIds;
  const GradInfos &gradInfos;

public:
  RequiredIds(std::set<TensorId> &tIds_, const GradInfos &gradInfos_)
      : tIds(tIds_), gradInfos(gradInfos_) {}

  void insert(const TensorId &tId) { tIds.insert(tId); }

  const GradInfo &gradInfo(SubGraphId sgId) const {
    return gradInfos.at(sgId);
  }

  SubGraphId grad(OpId opId, CalleeIndex ci) const {
    return gradInfos.grad(opId, ci);
  }
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
