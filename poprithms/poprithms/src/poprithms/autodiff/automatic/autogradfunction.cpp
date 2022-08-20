// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/automatic/autogradfunction.hpp>
#include <poprithms/autodiff/automatic/differentiator.hpp>
#include <poprithms/autodiff/core/autodiff.hpp>
#include <poprithms/program/callstack/calleetensorid.hpp>
#include <poprithms/program/callstack/carriedtensorid.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

void AutogradFunction::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

TensorIds AutogradFunction::apply(const TensorIds &insInCaller,
                                  const std::string &dbgName) {

  // Design note:
  //
  // If we want to re-use the forwards and backwards graphs, the current
  // approach is for the user to wrap their call to apply in a call op, in the
  // same way they would to reuse code for any operation.
  //
  // This does result in an inlining opportunity, where the user's call passes
  // straight through to the call generated here. The user's call can be
  // removed by an inlining optimization pass.
  //
  // We could do it automatically here. That is, we could automatically reuse
  // fwd-bwd where the signatures of #insInCaller are the same at multiple
  // call sites. This would require some kind of signature (virtual) method to
  // (1) check if 2 vectors of TensorIds are equivalent and (2) to hash it.
  //
  // This approach seems too specific to this project though, and I don't want
  // to invest in work which isn't general and reusable.
  //
  // Thus I think the sustainable approach to take is to implement a general
  // inliner, or rely on an inliner lower down the stack.

  auto &m = ad_.mutator();

  auto sgRootName_ = dbgName.empty() ? "atgd-fn" : dbgName;

  if (insInCaller.empty()) {
    throw poprithms::test::error(
        "At least one input required to AutoGrad forward function. dbgName=" +
        sgRootName_);
  }

  const auto nFwdIns = insInCaller.size();

  auto sgFwdId = m.createSubGraphId(sgRootName_ + "-fwd");
  auto sgBwdId = m.createSubGraphId(sgRootName_ + "-bwd");

  // The tensors in sgFwd which are inputs (copied to).
  TensorIds insInFwdCallee;
  insInFwdCallee.reserve(nFwdIns);
  for (auto tInCaller : insInCaller) {
    auto tInCallee =
        m.variableLike(tInCaller, sgFwdId, "like-" + tInCaller.str());
    insInFwdCallee.push_back(tInCallee);
  }

  TensorIds outsInFwdCallee = forwards(insInFwdCallee);
  const auto nFwdOuts       = outsInFwdCallee.size();

  // Create a backwards graph.
  core::GradInfo::CheckpointPairs cpIns;
  cpIns.reserve(nFwdOuts);
  TensorIds cpsInBwd;
  cpsInBwd.reserve(nFwdOuts);
  for (auto inFwdCallee : outsInFwdCallee) {
    auto cpIn =
        m.variableLike(inFwdCallee, sgBwdId, "cp-like-" + inFwdCallee.str());
    cpsInBwd.push_back(cpIn);
    cpIns.push_back({inFwdCallee, cpIn});
  }

  // 2) For all of the outputs in sgFwd, is there going to be a gradient
  //    provided for it in the backwards pass?
  core::GradInfo::GradInPairs gradIns;
  OptionalTensorIds gradInsToBwd;
  for (OutIndex o = 0; o < outsInFwdCallee.size(); ++o) {
    if (fwdOutGradUsedInBackwards(o)) {
      auto fwdOut = outsInFwdCallee.at(o.get());
      auto gIn = m.variableLike(fwdOut, sgBwdId, "grad-like-" + fwdOut.str());
      gradIns.push_back({fwdOut, gIn});
      gradInsToBwd.push_back(gIn);
    } else {
      gradInsToBwd.push_back({});
    }
  }

  auto bwdGrads = backwards(cpsInBwd, gradInsToBwd);

  if (bwdGrads.size() != nFwdIns) {
    std::ostringstream oss;
    oss << "The number of optional tensors returned by backwards is "
        << bwdGrads.size() << ", and the number of inputs to forwards is "
        << insInCaller.size() << ". " << bwdGrads.size()
        << " != " << insInCaller.size() << ".";
    throw error(oss.str());
  }

  core::GradInfo::TargetAndGradPairs finals;
  for (InIndex i = 0; i < bwdGrads.size(); ++i) {
    auto opt = bwdGrads.at(i.get());
    if (opt.has_value()) {
      auto dt = opt.value();
      finals.push_back({insInFwdCallee.at(i.get()), dt});
    }
  }

  auto gInfo = GradInfo::outOfGraph(sgFwdId, sgBwdId, gradIns, cpIns, finals);

  ad_.insertGradInfo(gInfo);

  const auto &q       = ad_.querier();
  SubGraphId sgCaller = q.subGraphId(insInCaller[0]);

  std::vector<std::pair<TensorId, TensorId>> fwdInputPairs;
  for (InIndex i = 0; i < insInCaller.size(); ++i) {
    fwdInputPairs.push_back(
        {insInCaller.at(i.get()), insInFwdCallee.at(i.get())});
  }

  // A call into the user defined forward pass.
  auto fwdCall = m.call(sgCaller, sgFwdId, fwdInputPairs, outsInFwdCallee);

  // Register the user defined backwards computation.
  ad_.setGrad(fwdCall, CalleeIndex(0), sgBwdId);

  TensorIds outs;
  for (auto x : outsInFwdCallee) {
    outs.push_back(
        q.dstInCaller(x, CallEvent(fwdCall, sgFwdId, CalleeIndex(0))));
  }

  return outs;
}

} // namespace automatic
} // namespace autodiff
} // namespace poprithms
