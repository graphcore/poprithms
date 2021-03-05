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
#include <poprithms/memory/unwind/subgraphid.hpp>
#include <poprithms/memory/unwind/valuedtensorid.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

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
          const SubGraphId sgid_,
          const std::vector<ValuedTensorIds> &valuedPartners_)
        : baseState(state), sgid(sgid_), valuedPartners(valuedPartners_) {}

    State(const OpId id_,
          const TensorIds &inIds_,
          const std::vector<ConsumptionIds> &consumptionIds_,
          const Shapes &inShapes_,
          const Shapes &outShapes_,
          const std::string &name_,
          const SubGraphId sgid_,
          const std::vector<ValuedTensorIds> &valuedPartners_)
        : State(common::multiout::Op::State(id_,
                                            inIds_,
                                            consumptionIds_,
                                            inShapes_,
                                            outShapes_,
                                            name_),
                sgid_,
                valuedPartners_) {}

    const common::multiout::Op::State baseState;

    // The SubGraphId of this Op.
    const SubGraphId sgid;

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
      : common::multiout::Op(ob.baseState), sgid_(ob.sgid),
        valuedPartners_(ob.valuedPartners) {}

  void insertAttractor(OutIndex, const TensorId &, double);

  SubGraphId subGraphId() const { return sgid_; }
  const std::vector<ValuedTensorIds> &valuedPartners() const;
  ValuedTensorIds valuedPartners(OutIndex outIndex) const;

  State getState() const;

  static State getStartingState(OpId,
                                SubGraphId,
                                const TensorIds &tensorIns,
                                const Shapes &inShapes,
                                const Shapes &outShapes);

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
  SubGraphId sgid_;
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
};

std::ostream &operator<<(std::ostream &, const Op &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
