// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <ostream>

#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/memory/inplace/consumer.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

void Consumer::append(std::ostream &ost) const {
  ost << "(" << opId() << "," << inIndex() << ')';
}

std::string Consumer::str() const {
  std::ostringstream oss;
  oss << *this;
  return oss.str();
}

std::ostream &operator<<(std::ostream &ost, const Consumer &c) {
  c.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const Consumers &consumers) {

  std::vector<std::string> frags;
  frags.reserve(consumers.size());
  for (const auto &c : consumers) {
    frags.push_back(c.str());
  }

  poprithms::util::append(ost, frags);
  return ost;
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
