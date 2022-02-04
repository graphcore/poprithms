// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <string>

#include <testutil/memory/unwind/fullstate.hpp>
#include <testutil/memory/unwind/op.hpp>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace unwindtoy {

void Op::removeSchedulableDerivedOutputs(const ContiguousOutIndexSubset &) {
  throw poprithms::test::error(
      "this test class does not support any transformations");
}

[[noreturn]] void invalid(const std::string &e = {}) {
  throw poprithms::test::error("Unimplemented method called. Context:" + e);
}

using State = poprithms::common::schedulable::Op::State;

// This class is just for testing, so we're not going to support Graph or Op
// comparison.
bool Op::schedulableTypeSpecificEqualTo(
    const poprithms::common::schedulable::Op &) const {
  invalid("schedulableTypeSpecificEqualTo");
}

void Op::growUnwind(FullState &u) const {

  // get the ids of the tensors in the unwind::Graph, and them register them
  // to their unwindtoy::Graph:
  auto outs = grow(u);
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    u.insert(outTensorId(o), outs.at(o));
    u.uwGraph().setName(outs.at(o).opId(),
                        "unwind equivalent of " + str() + ":" + getName());
  }
}

////////////
// MatMul //
////////////
std::unique_ptr<MultioutOp> MatMul::cloneMultioutOp() const {
  return std::make_unique<MatMul>(getSchedulableState(), atts_);
}

void MatMul::fwd(FullState &fs) const {

  // layout does not depend on inputs. Using the output unwind id to be
  // consistent  with createMappedSrc.
  auto outLayout = HTensor::uniformFloat32(
      -1, 1, outShape(0), fs.toUnwind(outTensorId(0)).opId().get());
  fs.setMainLayout(outTensorId(0),
                   fs.createMappedSrc(fs.toUnwind(outTensorId(0))));
}

TensorIds MatMul::grow(FullState &u) const {
  const auto uw0 = u.toUnwind(inTensorId(0));
  const auto uw1 = u.toUnwind(inTensorId(1));

  // lhs input creator
  auto lhs0 = u.uwGraph().barrier({}, {inShape(0)}, "lhs_" + str());

  // rhs input creator
  auto rhs0 = u.uwGraph().barrier({}, {inShape(1)}, "rhs_" + str());

  // the output, whose layout does not depend on inputs
  TensorIds mmindeps{};
  const auto out =
      u.uwGraph().barrier(mmindeps, {outShape(0)}, "mm_out_" + str());

  u.uwGraph().insertValuedPair({lhs0, 0}, uw0, atts_.lhs());
  u.uwGraph().insertValuedPair({rhs0, 0}, uw1, atts_.rhs());

  // points for matching the lhs input's layout to the outputs
  if (inShape(0) == outShape(0)) {
    u.uwGraph().insertValuedPair({out, 0}, uw0, atts_.lhsOut());
  }

  if (inShape(1) == outShape(0)) {
    u.uwGraph().insertValuedPair({out, 0}, uw1, atts_.rhsOut());
  }
  return {{out, 0}};
}

///////////
// slice //
///////////
std::unique_ptr<MultioutOp> Slice::cloneMultioutOp() const {
  return std::make_unique<Slice>(getSchedulableState(), lower_, upper_);
}

TensorIds Slice::grow(FullState &u) const {
  auto o = u.uwGraph().slice(u.toUnwind(inTensorId(0)), lower_, upper_);
  return {o};
}

void Slice::fwd(FullState &fs) const {
  fs.setMainLayout(outTensorId(0),
                   fs.mainLayout(inTensorId(0)).slice_(lower_, upper_));
}

std::string Slice::typeString() const {
  std::ostringstream oss;
  oss << "Slice_(l=";
  poprithms::util::append(oss, lower_);
  oss << ",u=";
  poprithms::util::append(oss, upper_);
  oss << ')';
  return oss.str();
}

/////////
// Sum //
/////////
std::unique_ptr<MultioutOp> Sum::cloneMultioutOp() const {
  return std::make_unique<Sum>(getSchedulableState(), unwindables_, sassy);
}

void Sum::fwd(FullState &fs) const {
  // this one is a bit weird. Just taking the first
  // unwind index. Poplar chooses the input with the best
  // tile spread.
  auto x = [this, &fs]() {
    if (!unwindables_.empty()) {
      return fs.mainLayout(inTensorId(unwindables_[0]));
    }

    // if there is no unwinding, generate a random layout.
    return HTensor::uniformFloat32(-1, +1, outShape(0), id().get());
  }();

  fs.setMainLayout(outTensorId(0), x);
}

TensorIds Sum::grow(FullState &u) const {
  auto o =
      u.uwGraph().sumLike(u.toUnwinds(inTensorIds()), unwindables_, sassy);
  return {o.out()};
}

////////////////
// DimShuffle //
////////////////
std::unique_ptr<MultioutOp> DimShuffle::cloneMultioutOp() const {
  return std::make_unique<DimShuffle>(getSchedulableState(), p_);
}

TensorIds DimShuffle::grow(FullState &u) const {
  auto o = u.uwGraph().dimShuffle(u.toUnwind(inTensorId(0)), p_);
  return {o};
}

void DimShuffle::fwd(FullState &fs) const {
  fs.setMainLayout(outTensorId(0),
                   fs.mainLayout(inTensorId(0)).dimShuffle_(p_));
}

std::string DimShuffle::typeString() const {
  return "DimShuffle(p=" + p_.str() + ")";
}

////////////
// Expand //
////////////
std::unique_ptr<MultioutOp> Expand::cloneMultioutOp() const {
  return std::make_unique<Expand>(getSchedulableState());
}

TensorIds Expand::grow(FullState &u) const {
  auto o = u.uwGraph().expand(u.toUnwind(inTensorId(0)), outShape(0));
  return {o};
}

void Expand::fwd(FullState &fs) const {
  fs.setMainLayout(outTensorId(0),
                   fs.mainLayout(inTensorId(0)).expand_(outShape(0)));
}

////////////
// Reduce //
////////////

std::unique_ptr<MultioutOp> Reduce::cloneMultioutOp() const {
  return std::make_unique<Reduce>(getSchedulableState());
}

TensorIds Reduce::grow(FullState &u) const {
  auto o = u.uwGraph().barrier({u.toUnwind(inTensorId(0))}, {outShape(0)});
  return TensorIds{{o, 0}};
}

void Reduce::fwd(FullState &fs) const {
  fs.setMainLayout(outTensorId(0),
                   fs.createMappedSrc(fs.toUnwind(outTensorId(0))));
}

std::string Expand::typeString() const { return "Expand"; }

///////////
// Input //
///////////
std::unique_ptr<MultioutOp> Input::cloneMultioutOp() const {
  return std::make_unique<Input>(getSchedulableState(), linear_);
}

void Input::fwd(FullState &fs) const {
  // get the layout from the unwindSinks_.

  // throw poprithms::test::error("Input::fwd should not be called");
  auto prepared = fs.getUnwindSink(fs.toUnwind(outTensorId(0)));

  for (auto x : prepared.getFloat32Vector()) {
    if (x - unmappedValue == 0.0) {
      throw poprithms::test::error("unmapped value detected in Input::fwd");
    }
  }
  fs.setMainLayout(outTensorId(0), prepared);
}

TensorIds Input::grow(FullState &u) const {
  auto o = u.uwGraph().sink(outShape(0));
  TensorId s{
      u.uwGraph().barrier(
          {}, {outShape(0)}, "linear mapper for " + str() + ":" + getName()),
      0};
  u.uwGraph().insertValuedPair(o, s, linear_);
  return {o};
}

std::unique_ptr<MultioutOp> Concat::cloneMultioutOp() const {
  return std::make_unique<Concat>(getSchedulableState(), axis_);
}

////////////
// Concat //
////////////
TensorIds Concat::grow(FullState &u) const {
  auto o = u.uwGraph().concat(u.toUnwinds(inTensorIds()), axis_);
  return {o};
}

void Concat::fwd(FullState &fs) const {
  HTensors ins_;
  for (auto inTensorId : inTensorIds()) {
    ins_.push_back(fs.mainLayout(inTensorId));
  }
  fs.setMainLayout(outTensorId(0), HTensor::concat_(ins_, axis_));
}

std::string Concat::typeString() const {
  return "Concat(axis=" + std::to_string(axis_) + ")";
}

} // namespace unwindtoy
} // namespace poprithms
