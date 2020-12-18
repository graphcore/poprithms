// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <typeinfo>

#include <poprithms/memory/inplace/crosslink.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

bool RegsMap::operator==(const RegsMap &rhs) const {
  // Same derived class, and same derived properties:
  return typeid(*this) == typeid(rhs) && typeSpecificEqualTo(rhs);
}

bool CrossLink::operator==(const CrossLink &rhs) const {
  return tup() == rhs.tup() && regsMap_.rm_->operator==(*rhs.regsMap_.rm_);
}

void CrossLink::append(std::ostream &ost) const {
  ost << in() << "->" << out();
  if (isModifying()) {
    ost << "[modifying]";
  } else if (isAliasing()) {
    ost << "[aliasing]";
  } else {
    ost << "[using]";
  }
}

// TODO(T31845) Unify this class's appearances as a utility.
CrossLink::WrappedRegsMap &
CrossLink::WrappedRegsMap::operator=(const WrappedRegsMap &rhs) {
  if (this == &rhs) {
    return *this;
  }
  this->rm_ = rhs.rm_->clone();
  return *this;
}

CrossLink::WrappedRegsMap::WrappedRegsMap(WrappedRegsMap &&rhs)
    : rm_(std::move(rhs.rm_)) {}

CrossLink::WrappedRegsMap &
CrossLink::WrappedRegsMap::operator=(WrappedRegsMap &&rhs) {
  if (this == &rhs) {
    return *this;
  }
  this->rm_ = std::move(rhs.rm_);
  return *this;
}

std::unique_ptr<RegsMap> IdentityRegsMap::clone() const {
  return std::make_unique<IdentityRegsMap>();
}

CrossLink CrossLink::modifies(InIndex i, OutIndex o) {
  return CrossLink(i, o, Type::Modifies, std::make_unique<IdentityRegsMap>());
}

CrossLink CrossLink::pureAliases(InIndex i, OutIndex o) {
  return CrossLink(
      i, o, Type::PureAliases, std::make_unique<IdentityRegsMap>());
}

CrossLink
CrossLink::uses(InIndex i, OutIndex o, std::unique_ptr<RegsMap> regsMap) {
  return CrossLink(i, o, Type::Uses, std::move(regsMap));
}

std::ostream &operator<<(std::ostream &ost, const CrossLinks &m) {
  ost << '(';
  if (!m.empty()) {
    m[0].append(ost);
  }
  for (uint64_t i = 1; i < m.size(); ++i) {
    ost << ',';
    m[i].append(ost);
  }
  ost << ')';
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const CrossLink &ca) {
  ca.append(ost);
  return ost;
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
