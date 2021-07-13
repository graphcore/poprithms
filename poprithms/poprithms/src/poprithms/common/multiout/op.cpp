// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <common/multiout/error.hpp>
#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <poprithms/common/multiout/op.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace common {
namespace multiout {

Op::State::State(const OpId id_,
                 const TensorIds &inIds_,
                 const std::vector<ConsumptionIds> &consumptionIds_,
                 const Shapes &inShapes_,
                 const Shapes &outShapes_,
                 const std::string &name_)
    : id(id_), inIds(inIds_), consumptionIds(consumptionIds_),
      inShapes(inShapes_), outShapes(outShapes_), name(name_) {

  if (inIds.size() != inShapes.size()) {
    std::ostringstream oss;
    oss << "The number of input Shapes should be the same as "
        << "the number of input Ids, "
        << "in the multiout::Op::State constructor. "
        << "This for State with id=" << id_ << " and name=\"" << name_
        << "\", "
        << ", where the number of input Shapes is " << inShapes.size()
        << " but the number of input ids is " << inIds.size() << ".";
    throw error(oss.str());
  }

  if (consumptionIds.size() != outShapes.size()) {
    std::ostringstream oss;
    oss << "The size of the vectors consumptionIds and outShapes "
        << "should be the same in the multiout::Op::State constructor. "
        << "But consumptionIds is of size " << consumptionIds.size()
        << ", and outShapes is of size " << outShapes.size() << ". "
        << "They should both be exactly equal to the number of outputs. ";
    throw error(oss.str());
  }
}

Op::~Op() = default;

Op::Op(const State &ob)
    : id_(ob.id), inIds_(ob.inIds), consumptionIds_(ob.consumptionIds),
      inShapes_(ob.inShapes), outShapes_(ob.outShapes), name_(ob.name) {}

// C++20 we will be able to just use (= default).
bool Op::State::operator==(const State &rhs) const {
  return                                      //
      id == rhs.id &&                         //
      inIds == rhs.inIds &&                   //
      consumptionIds == rhs.consumptionIds && //
      inShapes == rhs.inShapes &&             //
      outShapes == rhs.outShapes &&           //
      name == rhs.name;
}

TensorIds Op::outTensorIds() const {
  TensorIds outIds;
  outIds.reserve(nOutTensors());
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    outIds.push_back(TensorId{id(), OutIndex(o)});
  }
  return outIds;
}

std::ostream &operator<<(std::ostream &os, const Op &op) {
  os << op.str();
  if (!op.getName().empty()) {
    os << "::" << op.getName();
  }
  return os;
}

void Op::verify(InIndex inIndex,
                OutIndex outIndex,
                const std::string &context) const {
  if (inIndex >= nInTensors()) {
    std::ostringstream oss;
    oss << "Invalid InIndex in " << context << " of " << typeString()
        << ". InIndex=" << inIndex << ", but nInTensors=" << nInTensors()
        << '.';
    throw error(oss.str());
  }
  if (outIndex >= nOutTensors()) {
    std::ostringstream oss;
    oss << "Invalid OutIndex in " << context << " of " << typeString()
        << ". InIndex=" << inIndex << ", but nOutTensors=" << nOutTensors()
        << '.';
    throw error(oss.str());
  }
}

std::string Op::str() const {
  return typeString() + std::string("::") + id();
}

bool Op::operator==(const Op &rhs) const {
  return
      // Same common properties:
      getState() == rhs.getState() &&
      // Same derived class:
      typeid(*this) == typeid(rhs) &&
      // Same derived class properties:
      multiOutTypeSpecificEqualTo(rhs);
}

} // namespace multiout
} // namespace common
} // namespace poprithms
