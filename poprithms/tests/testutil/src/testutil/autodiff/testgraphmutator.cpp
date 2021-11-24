// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <sstream>

#include <testutil/autodiff/testgraphmutator.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace autodiff {
namespace test {

OpId TestGraphMutator::clone(OpId id, const TensorIds &ins) {
  auto toClone = c.op(id);
  return c.insert(Op(ins,
                     toClone.nOuts,
                     toClone.insRequired,
                     toClone.outsRequired,
                     toClone.flows,
                     "clone of " + std::to_string(id.get()),
                     toClone.type));
}
TensorId TestGraphMutator::add(const TensorId &t0, const TensorId &t1) {
  return {c.insert(Op(
              {t0, t1}, 1, {}, {}, {{1, 0}, {0, 0}}, "Add", Op::Type::Add)),

          0};
}

OptionalTensorIds
TestGraphMutator::getInGrads(OpId opId,
                             const core::ToGradGraph &toGradGraph) {

  auto fwdOp = c.op(opId);

  // the output indices which can propagate to one or more inputs. these
  // must have gradients available.
  std::vector<OutIndex> providedGrads;
  for (uint64_t o = 0; o < c.nOutTensors(opId); ++o) {
    if (std::any_of(fwdOp.flows.cbegin(),
                    fwdOp.flows.cend(),
                    [o](const auto &f) { return f.o == OutIndex(o); })) {
      providedGrads.push_back(OutIndex(o));
    }
  }

  TensorIds gradOpIns;
  for (auto o : providedGrads) {
    gradOpIns.push_back(toGradGraph.getGrad({opId, o}));
  }
  for (auto inReq : fwdOp.insRequired) {
    gradOpIns.push_back(toGradGraph.getNonGrad(c.inTensorId(opId, inReq)));
  }
  for (auto outReq : fwdOp.outsRequired) {
    gradOpIns.push_back(toGradGraph.getNonGrad({opId, outReq}));
  }

  // the input indices of opId to which gradients can propagate from the
  // outputs.
  std::vector<InIndex> requireGrad;
  for (uint64_t i = 0; i < c.nInTensors(opId); ++i) {
    if (std::any_of(fwdOp.flows.cbegin(),
                    fwdOp.flows.cend(),
                    [i](const auto &f) { return f.i == InIndex(i); })) {
      requireGrad.push_back(InIndex(i));
    }
  }

  auto type = Op::grad(fwdOp.type);

  auto gOp = c.insert(Op(gradOpIns,
                         requireGrad.size(),
                         {},
                         {},
                         {},
                         "grad-of-" + std::to_string(opId.get()),
                         type));

  OptionalTensorIds opts(c.nInTensors(opId));
  for (uint64_t i = 0; i < requireGrad.size(); ++i) {
    opts[requireGrad.at(i).get()] = TensorId(gOp, i);
  }

  return opts;
}

} // namespace test
} // namespace autodiff
} // namespace poprithms
