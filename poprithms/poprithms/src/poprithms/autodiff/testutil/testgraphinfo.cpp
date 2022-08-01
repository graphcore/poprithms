// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <sstream>

#include <poprithms/autodiff/testutil/testgraphinfo.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace autodiff {
namespace testutil {

OpId TestGraphInfo::insert(Op op_) {

  ops.push_back(op_);
  OpId nxt = ops.size() - 1;
  for (uint64_t i = 0; i < op_.ins.size(); ++i) {
    TensorId in_ = op_.ins[i];
    op(in_.opId())
        .consumers[in_.outIndex().get()]
        .push_back(ConsumptionId(nxt, i));
  }
  return nxt;
}

TensorId TestGraphInfo::insertNoFlow(const TensorIds &ins_,
                                     const std::string &n,
                                     Op::Type t) {
  auto opId = insert(Op(ins_, 1, {}, {}, {}, n, t));
  return {opId, 0};
}

OpId TestGraphInfo::matmul(const TensorId &a,
                           const TensorId &b,
                           const std::string &n) {
  return insert(
      Op({a, b}, 1, {0, 1}, {}, {{0, 0}, {0, 1}}, n, Op::Type::Matmul));
}

bool TestGraphInfo::gradientPropagates(const OpTraversal &ot) const {

  const auto flows = ops.at(ot.opId().get()).flows;
  const auto t     = Op::Flow(ot);
  const auto props = std::any_of(flows.cbegin(),
                                 flows.cend(),
                                 [t](const Op::Flow &o) { return o == t; });

  return props;
}

void TestGraphInfo::extendAutodiffRequiredTensors(
    OpId id,
    std::set<TensorId> &ids) const {
  for (auto i : op(id).insRequired) {
    ids.insert(op(id).ins[i.get()]);
  }

  for (auto o : op(id).outsRequired) {
    ids.insert({id, OutIndex(o)});
  }
}

void TestGraphInfo::assertCanBeRerun(OpId id, bool) const {
  if (nInTensors(id) == 0) {
    throw poprithms::test::error(
        "var creators (ops without inputs) cannot be rerun");
  }
}

std::ostream &operator<<(std::ostream &ost, const TestGraphInfo &controler) {
  controler.append(ost);
  return ost;
}

template <typename T>
auto add(std::vector<std::string> &x, const std::vector<T> &y) {
  std::ostringstream ost;
  poprithms::util::append(ost, y);
  x.push_back(ost.str());
}

std::ostream &operator<<(std::ostream &ost, const Op::Flow &f) {
  ost << f.i << "<-" << f.o;
  return ost;
}

void TestGraphInfo::append(std::ostream &ost) const {

  using namespace poprithms::util;

  std::vector<StringColumn> columns;

  std::vector<std::string> ids_;
  std::vector<std::string> ins_;
  std::vector<std::string> nOuts_;
  std::vector<std::string> insRequired_;
  std::vector<std::string> outsRequired_;
  std::vector<std::string> flows_;
  std::vector<std::string> names_;
  std::vector<std::string> types_;

  for (uint64_t opId = 0; opId < nOps(); ++opId) {
    const auto &op = ops.at(opId);
    ids_.push_back(std::to_string(opId));
    add(ins_, op.ins);
    nOuts_.push_back(std::to_string(op.nOuts));
    add(insRequired_, op.insRequired);
    add(outsRequired_, op.outsRequired);
    add(flows_, op.flows);
    names_.push_back(op.name);
    types_.push_back(Op::str(op.type));
  }

  columns.push_back({"Id", ids_, {}});
  columns.push_back({"Type", types_, {}});
  columns.push_back({"Ins", ins_, {}});
  columns.push_back({"nOut", nOuts_, {}});
  columns.push_back({"insRequired", insRequired_, {}});
  columns.push_back({"outsRequired", outsRequired_, {}});
  columns.push_back({"flows", flows_, {}});
  columns.push_back({"name", names_, {}});

  ost << alignedColumns(columns);
}

} // namespace testutil
} // namespace autodiff
} // namespace poprithms
