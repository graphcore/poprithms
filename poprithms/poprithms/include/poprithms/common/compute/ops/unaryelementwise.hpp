// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_UNARYELEMENTWISE_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_UNARYELEMENTWISE_HPP

#include <exception>
#include <memory>

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withautodiff.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/common/multiout/ioindices.hpp>

namespace poprithms {
namespace common {
namespace compute {

using common::multiout::InIndices;
using common::multiout::OutIndices;

using autodiff::automatic::ZeroPropagationAutodiffer;

/**
 * An elementwise op with 1 input and 1 output.
 * */
class UnaryElementwise : public WithoutCalleesTensorCentric {
public:
  UnaryElementwise(const State &s) : WithoutCalleesTensorCentric(s) {}

private:
  /**
   * A unary elementwise op does computation, and is therefore not an
   * 'initializing op'.
   * */
  bool isInitializingOp() const final { return false; }

  void runSim(ISimState &ss) const final {
    runReplicatedSim(ss.simTensorMap());
  }

  CodeLocation codeLocation() const final { return locationByUnanimity(); }

  /**
   * The output is not a reference to a tensor in another graph. (\sa the
   * RefFrom op class).
   * */
  void resetRootRef(OutIndex, const TensorId &) { invalid(); }
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  /**
   * Unary elementwise ops have 1 input and 1 output. They need only therefore
   * define how the host computation is done with 1 input and 1 output.
   * */
  void compute(const HostTensors &ins, const HostTensors &outs) const final {
    unaryCompute(ins[0], outs[0]);
  }

  /**
   * Update the value of #outTensor by performing this elementwise operation
   * on #inTensor.
   * */
  virtual void unaryCompute(const HostTensor &inTensor,
                            const HostTensor &outTensor) const = 0;

  void computeDerivedVerifyValid() const final;

protected:
  /// For the verifier, this enum defines the expected output type.
  enum class OutType {
    Preserving, ///< Output type <- Input type.
    Bool,       ///< Output type <- Boolean.
    Other       ///< Output type should not be checked.
  };

private:
  // As a final check of the method #computeDerivedVerifyValid, this method is
  // called. Ops can optionally (it is not pure virtual) add additional checks
  // on attributes in this method.
  virtual void unaryElementwiseDerivedVerifyValid() const {}
  virtual OutType outType() const = 0;
};

/**
 * An inplace unary elementwise op. The output is an alias of the input.
 * */
class UnaryElementwiseInplace_ : public UnaryElementwise {
public:
  UnaryElementwiseInplace_(const State &s) : UnaryElementwise(s) {}
  bool aliases(InIndex, OutIndex) const final { return true; }
  bool modifies(InIndex) const final { return true; }

  /**
   * Create a new variable/allocation in the alias::Graph corresponding to the
   * output of this op.
   * */
  void growAliasMapper(MemoryAliasMapper &mam) const final {
    createAlias(mam, inTensorId(0));
  }

  HostTensors initializeOut(const HostTensors &) const final;

protected:
  // A method which is used by non-monotonic ops which cannot be
  // backpropagated through.
  std::string nonMonotonicInplace() const {
    return "Non-monotonic inplace unary ops like " + str() +
           " cannot be backpropagated through as their inputs are required "
           "but not available as they get written to and cannot be be "
           "recomputed from the output. ";
  }

private:
  // The type of the output of an inplace unary op must be the same as the
  // input tensor's type (as it is the same tensor).
  OutType outType() const final { return OutType::Preserving; }
};

/**
 * Fill all the elements of the input tensor with a constant scalar value.
 * */
class Fill_ final : public WithAutodiff<ZeroPropagationAutodiffer,
                                        UnaryElementwiseInplace_> {

public:
  Fill_(const State &s, const HostTensor &v) : WithAutodiff(s), val_(v) {}

  /**
   * The scalar value which the input tensor is filled with.
   * */
  HostTensor value() const { return val_; }

private:
  std::string typeString() const final;
  UpOp cloneWithState(const State &s) const final;
  bool computeTypeSpecificEqualTo(const Op &) const final;

  void unaryCompute(const HostTensor &, const HostTensor &out) const final {
    out.update_(val_);
  }

  // verify that val_ has only 1 element.
  void unaryElementwiseDerivedVerifyValid() const final;

  HostTensor val_;
};

/**
 * This unary elementwise op's output is not an alias of its input.
 * */
class UnaryElementwiseOutplace : public UnaryElementwise {
public:
  UnaryElementwiseOutplace(const State &s) : UnaryElementwise(s) {}

private:
  void growAliasMapper(MemoryAliasMapper &mam) const final {
    // create new variables for the outputs.
    createVariables(mam);
  }

  bool aliases(InIndex, OutIndex) const final { return false; }
  bool modifies(InIndex) const final { return false; }
  HostTensors initializeOut(const HostTensors &) const final;
};

class Cast final : public Attributeless<UnaryElementwiseOutplace, Cast> {
public:
  Cast(const State &s) : Attributeless(s) {}

  // This static char * is required by the Attributeless template class, which
  // implemenents a virtual method using it:
  static constexpr const char *OpTypeName = "Cast";

private:
  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }

  /**
   * Gradient propagates if neither the input nor the output types are fixed
   * point. In other words, they must both be floating point tensors.
   * */
  bool gradientPropagates(OutIndex, InIndex) const final;

  OptionalTensors bprop(const GradOpIns &) const final;

  void unaryCompute(const HostTensor &, const HostTensor &) const final;

  OutType outType() const final { return OutType::Other; }
};

// An outplace op which does not propagate a non-zero gradient to the input.
using ZeroGradUnaryElementwiseOutplace =
    WithAutodiff<ZeroPropagationAutodiffer, UnaryElementwiseOutplace>;

// An inplace op which does not propagate a non-zero gradient to the input.
using ZeroGradUnaryElementwiseInplace_ =
    WithAutodiff<ZeroPropagationAutodiffer, UnaryElementwiseInplace_>;

/**
 * This op computes the natural logrithm of its input. The template names in
 * the class it inherits from are:
 *  - Attributeless: says that Log has no additional attributes.
 *  - WithAutodiff<autodiff::automatic::LogAutodiffer...>: points to the class
 *                        which implements all of the autodiff logic for log.
 *
 * The 'curious' appearance of the class name 'Log' in the base class is
 * a C++ technique called CRTP.
 * */
class Log final
    : public Attributeless<WithAutodiff<autodiff::automatic::LogAutodiffer,
                                        UnaryElementwiseOutplace>,
                           Log> {

public:
  Log(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Log";

private:
  void unaryCompute(const HostTensor &i, const HostTensor &o) const final;
  OutType outType() const final { return OutType::Preserving; }
};

class Log_ final
    : public Attributeless<NoAutodiff<UnaryElementwiseInplace_>, Log_> {

public:
  Log_(const State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Log_";

private:
  void unaryCompute(const HostTensor &, const HostTensor &o) const final;
  std::string whyNoAutodiff() const final;
};

class Exp final
    : public Attributeless<WithAutodiff<autodiff::automatic::ExpAutodiffer,
                                        UnaryElementwiseOutplace>,
                           Exp> {

public:
  Exp(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Exp";

private:
  void unaryCompute(const HostTensor &i, const HostTensor &o) const final {
    o.update_(i.exp());
  }
  OutType outType() const final { return OutType::Preserving; }
};

class Exp_ final
    : public Attributeless<WithAutodiff<autodiff::automatic::ExpAutodiffer,
                                        UnaryElementwiseInplace_>,
                           Exp_> {

public:
  Exp_(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Exp_";

private:
  void unaryCompute(const HostTensor &i, const HostTensor &) const final {
    i.exp_();
  }
};

class Neg final
    : public Attributeless<WithAutodiff<autodiff::automatic::NegAutodiffer,
                                        UnaryElementwiseOutplace>,
                           Neg> {

public:
  Neg(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Neg";

private:
  void unaryCompute(const HostTensor &i, const HostTensor &o) const final {
    o.update_(i.neg());
  }
  OutType outType() const final { return OutType::Preserving; }
};

class Neg_ final
    : public Attributeless<WithAutodiff<autodiff::automatic::NegAutodiffer,
                                        UnaryElementwiseInplace_>,
                           Neg_> {

public:
  Neg_(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Neg_";

private:
  void unaryCompute(const HostTensor &i, const HostTensor &) const final {
    i.neg_();
  }
};

class Sqrt final
    : public Attributeless<WithAutodiff<autodiff::automatic::SqrtAutodiffer,
                                        UnaryElementwiseOutplace>,
                           Sqrt> {

public:
  Sqrt(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Sqrt";

private:
  void unaryCompute(const HostTensor &i, const HostTensor &o) const final {
    o.update_(i.sqrt());
  }
  OutType outType() const final { return OutType::Preserving; }
};

class Sqrt_ final
    : public Attributeless<WithAutodiff<autodiff::automatic::SqrtAutodiffer,
                                        UnaryElementwiseInplace_>,
                           Sqrt_> {

public:
  Sqrt_(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Sqrt_";

private:
  void unaryCompute(const HostTensor &i, const HostTensor &) const final {
    i.sqrt_();
  }
};

class Sin final : public Attributeless<UnaryElementwiseOutplace, Sin> {

public:
  static constexpr const char *OpTypeName = "Sin";
  Sin(const State &s) : Attributeless(s) {}

private:
  void unaryCompute(const HostTensor &i, const HostTensor &o) const final {
    o.update_(i.sin());
  }
  std::vector<InIndex> autodiffRequiredIns() const final { return {0}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
  bool gradientPropagates(OutIndex, InIndex) const final { return true; }

  /**
   * The gradient of sin is cos.
   * */
  OptionalTensors bprop(const GradOpIns &gIns) const final {
    return {gIns.input(0).cos() * gIns.gradOfOutput(0)};
  }
  OutType outType() const final { return OutType::Preserving; }
};

class Sin_ final
    : public Attributeless<NoAutodiff<UnaryElementwiseInplace_>, Sin_> {
public:
  Sin_(const State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Sin_";

private:
  void unaryCompute(const HostTensor &, const HostTensor &o) const final {
    o.sin_();
  }
  std::string whyNoAutodiff() const final { return nonMonotonicInplace(); }
};

class Abs final : public Attributeless<UnaryElementwiseOutplace, Abs> {
public:
  static constexpr const char *OpTypeName = "Abs";
  Abs(const State &s) : Attributeless(s) {}
  void unaryCompute(const HostTensor &i, const HostTensor &o) const final {
    o.update_(i.abs());
  }

private:
  std::vector<InIndex> autodiffRequiredIns() const final { return {0}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
  bool gradientPropagates(OutIndex, InIndex) const final { return true; }
  OptionalTensors bprop(const GradOpIns &gIn) const final {
    return {gIn.gradOfOutput(0) * gIn.input(0).signum()};
  }

  OutType outType() const final { return OutType::Preserving; }
};

class Abs_ final
    : public Attributeless<NoAutodiff<UnaryElementwiseInplace_>, Abs_> {
public:
  Abs_(const State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Abs_";

private:
  void unaryCompute(const HostTensor &, const HostTensor &o) const final {
    o.abs_();
  }
  std::string whyNoAutodiff() const final {
    return UnaryElementwiseInplace_::nonMonotonicInplace();
  }
};

class Cos final : public Attributeless<UnaryElementwiseOutplace, Cos> {
public:
  static constexpr const char *OpTypeName = "Cos";
  Cos(const State &s) : Attributeless(s) {}

private:
  void unaryCompute(const HostTensor &i, const HostTensor &o) const final {
    o.update_(i.cos());
  }
  std::vector<InIndex> autodiffRequiredIns() const final { return {0}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
  bool gradientPropagates(OutIndex, InIndex) const final { return true; }

  /**
   * The gradient of cos(x) is -sin(x).
   **/
  OptionalTensors bprop(const GradOpIns &gIns) const final {
    return {gIns.gradOfOutput(0).mul(gIns.input(0).sin()).neg()};
  }
  OutType outType() const final { return OutType::Preserving; }
};

class Cos_ final
    : public Attributeless<NoAutodiff<UnaryElementwiseInplace_>, Cos_> {
public:
  Cos_(const State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Cos_";

private:
  void unaryCompute(const HostTensor &, const HostTensor &o) const final {
    o.cos_();
  }

  std::string whyNoAutodiff() const final {
    return UnaryElementwiseInplace_::nonMonotonicInplace();
  }
};

/**
 * The sign of the input (+1 / 0 / -1). The output type is the same the input
 * type. This op propagates a zero gradient back to the input (i.e. it does
 * not propagate a non-zero gradient).
 * */
class Signum final
    : public Attributeless<ZeroGradUnaryElementwiseOutplace, Signum> {

public:
  Signum(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Signum";

private:
  void unaryCompute(const HostTensor &i, const HostTensor &o) const final {
    o.update_(i.sign());
  }

  // Note that the output type is NOT a small integer. It is the same as the
  // input type.
  OutType outType() const final { return OutType::Preserving; }
};

class Signum_ final
    : public Attributeless<ZeroGradUnaryElementwiseInplace_, Signum_> {
public:
  Signum_(const Op::State &s) : Attributeless(s) {}
  static constexpr const char *OpTypeName = "Signum_";

private:
  void unaryCompute(const HostTensor &i, const HostTensor &) const final {
    i.sign_();
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
