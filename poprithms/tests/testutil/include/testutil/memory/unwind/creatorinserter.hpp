// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_TESTUTIL_UNWIND_TOY_CREATORINSERTER_HPP
#define POPRITHMS_TESTUTIL_UNWIND_TOY_CREATORINSERTER_HPP

#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/memory/unwind/matmulattractions.hpp>
#include <poprithms/memory/unwind/scheduledsolution.hpp>

namespace poprithms {
namespace unwindtoy {

class FullState;

using MatMulAttractions = poprithms::memory::unwind::MatMulAttractions;
using common::multiout::ConsumptionId;
using common::multiout::ConsumptionIds;
using common::multiout::InIndex;
using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OutIndex;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using memory::unwind::ScheduledSolution;
using ndarray::Shape;
using ndarray::Shapes;
using util::Permutation;
using Lower         = ndarray::Shape::Lower;
using MultioutOp    = common::multiout::Op;
using Upper         = ndarray::Shape::Upper;
using HTensor       = compute::host::Tensor;
using HTensors      = std::vector<HTensor>;
using SchedulableOp = common::schedulable::Op;
using memory::unwind::Path;
using State = SchedulableOp::State;
using common::multiout::ContiguousInIndexSubset;
using common::multiout::ContiguousOutIndexSubset;

class MatMulTensorCreatorInserter {
public:
  void insertMatMulLhsCreator(const TensorId &) const {
    // for poplar backend, will call method to create lhs input.
  }
  void insertMatMulRhsCreator(const TensorId &) const {
    // for poplar backend, will call method to create rhs input.
  }
  void insertMatMulOutCreator(const TensorId &) const {
    // for poplar backend, will call method to create output tensor. will use
    // dummy program & dummy inputs. See comments in graph.hpp on matmul.
  }
  OpId opId() const { return 0; }
};

} // namespace unwindtoy
} // namespace poprithms

#endif
