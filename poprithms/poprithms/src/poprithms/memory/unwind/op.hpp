// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_OP_HPP
#define POPRITHMS_MEMORY_UNWIND_OP_HPP

#include <algorithm>
#include <memory>
#include <sstream>
#include <typeinfo>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/memory/unwind/valuedtensorid.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

class Graph;

using ContiguousOutIndexSubset = poprithms::util::ContiguousSubset<OutIndex>;
using chain::Chain;
using common::multiout::ConsumptionIds;
using common::multiout::TensorIds;
using ndarray::Shapes;

/** An Op in an unwinding Graph. */
class Op : public common::multiout::Op {

public:
  struct State {

  public:
    State(const common::multiout::Op::State &state,
          const std::vector<ValuedTensorIds> &valuedPartners_)
        : baseState(state), valuedPartners(valuedPartners_) {}

    State(const OpId id_,
          const TensorIds &inIds_,
          const std::vector<ConsumptionIds> &consumptionIds_,
          const Shapes &outShapes_,
          const std::string &name_,
          const std::vector<ValuedTensorIds> &valuedPartners_,
          const Graph &g_);

    const common::multiout::Op::State baseState;

    // Tensors which would benefit from having the same layout as this Tensor.
    const std::vector<ValuedTensorIds> valuedPartners;

    // Will be  "=default" in C++20, but for now must be done manually.
    bool operator==(const State &rhs) const;
  };

  virtual ~Op();
  Op &operator=(const Op &) = default;
  Op &operator=(Op &&) = default;
  Op(const Op &)       = default;
  Op(Op &&)            = default;
  Op()                 = delete;

  Op(const State &ob)
      : common::multiout::Op(ob.baseState),
        valuedPartners_(ob.valuedPartners) {}

  void insertAttractor(OutIndex, const TensorId &, double);

  const std::vector<ValuedTensorIds> &valuedPartners() const;
  ValuedTensorIds valuedPartners(OutIndex outIndex) const;

  State getState() const;

  static State getStartingState(OpId,
                                const TensorIds &tensorIns,
                                const Shapes &outShapes,
                                const Graph &);

  DisjointRegions
  outRegions(const DisjointRegions &, InIndex, OutIndex) const;
  DisjointRegions inRegions(const DisjointRegions &, InIndex, OutIndex) const;

  virtual void extendFwd(Chain &, InIndex, OutIndex) const = 0;
  virtual void extendBwd(Chain &, InIndex, OutIndex) const = 0;
  void extend(Chain &, InIndex, OutIndex, bool isFwd) const;

  virtual bool isSink(OutIndex) const                = 0;
  virtual bool isSource(OutIndex) const              = 0;
  virtual bool isUnwindable(InIndex, OutIndex) const = 0;
  virtual bool isBarrier(OutIndex) const             = 0;

  std::vector<InIndex> unwindableIndices(OutIndex) const;
  std::vector<OutIndex> unwindableIndices(InIndex) const;

protected:
  using UpBop = std::unique_ptr<common::multiout::Op>;
  template <typename OP> static UpBop mu(const OP *const derived) {
    return std::make_unique<OP>(*derived);
  }

private:
  std::vector<ValuedTensorIds> valuedPartners_;

  /**
   * A pure virtual function that derived classes must implement.
   * This function has a precondition that it will only
   * be called when the 'other' is the same type as the instance
   * invoking the function.
   * */
  virtual bool unwindTypeSpecificEqualTo(const Op &other) const = 0;

  bool
  multiOutTypeSpecificEqualTo(const common::multiout::Op &other) const final;

  virtual void
  removeMultioutDerivedOutputs(const ContiguousOutIndexSubset &) final {
    unimplemented();
  }
};

std::ostream &operator<<(std::ostream &, const Op &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
