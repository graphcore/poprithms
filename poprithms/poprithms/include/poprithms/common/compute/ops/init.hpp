// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_INIT_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_INIT_HPP

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * An op which has no inputs, and which initializes one tensor.
 * */
class Init : public WithoutCallees {
public:
  Init(const Op::State &s) : WithoutCallees(s) {}

  /**
   * The shape of the single output tensor being initialized.
   * */
  Shape shape() const { return outShape(0); }

  /**
   * Init ops do no computation, and is this sense they are 'initializing ops'
   * (as are view-changing ops, for example).
   * */
  bool isInitializingOp() const final { return true; }

  /**
   * Initializing ops do no computation.
   * */
  void runSim(ISimState &) const final {}

  /**
   * Initializing ops do no computation.
   *
   * Note that we do not check that constant tensors on host are not written
   * to (T63236).
   * */
  void compute(const HostTensors &, const HostTensors &) const final {}

  /**
   * Initializing ops do no computation, and therefore have no code location.
   * */
  CodeLocation codeLocation() const final { return CodeLocation::None; }

  /**
   * The output is not a derived reference to a tensor in a different
   * sub-graph.
   * */
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  /**
   * The root reference cannot be anything other that the output tensor of the
   * op, attempts to reset it result in an error.
   * */
  void resetRootRef(OutIndex, const TensorId &) final { invalid(); }

  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
  OptionalTensorIds backpropagate(Graph &, const GradOpInIds &) const final {
    return {};
  }

  /**
   * Calls to these methods are always invalid, as Init op has has no inputs
   * and so the InIndex passed to these methods is necessarily invalid.
   * */
  bool aliases(InIndex i, OutIndex) const final { invalidInIndex(i); }
  bool modifies(InIndex i) const final { invalidInIndex(i); }
  bool gradientPropagates(OutIndex, InIndex i) const final {
    invalidInIndex(i);
  }

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

private:
  [[noreturn]] void invalidInIndex(InIndex) const;

  bool isValueDependent(InIndex i, OutIndex) const final {
    invalidInIndex(i);
  }
};

/**
 * Initialize a constant (read-only) tensor.
 * */
class ConstInit final : public Init {
public:
  /**
   * \param value. The value of the constant. This tensor must have the shape
   *               and type of the single output in the State #s.
   *
   * The elements of value are not copied.
   *
   * <code>
   * auto v = HostTensor::int32(1);
   * ConstInit cop(state, v);
   * v.add_(1);
   * cop.value().assertAllEquivalent(HostTensor::int32(2)); // no error.
   * </code>
   *
   * Use v.copy() to copy the elements of #v to a new tensor.
   * */
  ConstInit(const State &s, const HostTensor &value);

  std::string typeString() const final;

  /**
   * Create a clone of this op, where the value of the clone is a deep copy if
   * #pointerOnly is false.
   * */
  UpOp cloneConstInitWithState(const State &, bool pointerOnly) const;

  /**
   * Create a clone of this op. The clone's value is a deep copy of this op's
   * value, so a change to one will be reflected in the other.
   * */
  UpOp cloneWithState(const State &s) const final {
    return cloneConstInitWithState(s, true);
  }

  HostTensor value() const { return value_; }

private:
  /**
   * Create a tensor in the alias::Graph of #mam whose allocation has a
   * 'Color' which encodes constness.
   * */
  void growAliasMapper(MemoryAliasMapper &mam) const final;

  HostTensors initializeOut(const HostTensors &) const final {
    return {value()};
  }

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }
  void computeDerivedVerifyValid() const final;

  /**
   * Numerically equivalent. (Same shape, type, and values (bitwise)).
   **/
  bool computeTypeSpecificEqualTo(const compute::Op &) const final;

  HostTensor value_;
};

class VarInit final : public Init {

public:
  VarInit(const State &s) : Init(s) {}

  std::string typeString() const final;

  /**
   * Create a clone of this VarInit op with the State #s.
   * */
  UpOp cloneWithState(const State &) const final;

  /**
   * VarInit ops on host can either be 'user managed' or not.
   *
   * User managed: the host tensor will be a wrapper around a raw pointer,
   * which the user mananges.
   *
   * Not user managed: the host tensor will manage its own memory by reference
   * counting (see the host::Tensor class for more information, compare
   * PointerData and AllocData and see the diagram accompanying BaseData).
   *
   * The advantage of 'user managed' is that there is potentially 1 fewer copy
   * of data.
   * */
  bool isUserManagedHost() const;
  void setUserManagedHost(bool isUserManaged);

private:
  bool computeTypeSpecificEqualTo(const compute::Op &) const final;

  /**
   * Initialize (and return) a host tensor for this op. The single host
   * tensor returned has the shape and type of this op's output.
   *
   * If this op is 'user managed', then the tensor is initialized with a
   * nullptr, which the user must set at a later time. Otherwise, the host
   * tensor is initialized with non-zero values.
   *
   * \sa isUserManagedHost
   * */
  HostTensors initializeOut(const HostTensors &) const final;

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  void computeDerivedVerifyValid() const final;

  /**
   * The equivalant of the output of a ConstInit op in an alias::Graph is an
   * allocation whose 'Color' encodes non-constness.
   * */
  void growAliasMapper(MemoryAliasMapper &b) const final;

  enum class UserManagedHost { No = 0, Yes };
  UserManagedHost userManagedHost_{UserManagedHost::No};
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
