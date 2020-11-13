// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_CONSUMER_HPP
#define POPRITHMS_MEMORY_INPLACE_CONSUMER_HPP

#include <ostream>

#include <poprithms/memory/inplace/usings.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/**
 * Description of an Op which consumes a Tensor, and the input index at which
 * the Tensor is consumed.
 * */
class Consumer {

public:
  Consumer() = delete;
  Consumer(OpId opId__, InIndex inIndex__)
      : opId_(opId__), inIndex_(inIndex__) {}

  bool operator==(const Consumer &rhs) const {
    return opId() == rhs.opId() && inIndex() == rhs.inIndex();
  }

  OpId opId() const { return opId_; }
  InIndex inIndex() const { return inIndex_; }
  void append(std::ostream &) const;
  std::string str() const;

private:
  OpId opId_;
  InIndex inIndex_;
};

using Consumers = std::vector<Consumer>;

std::ostream &operator<<(std::ostream &, const Consumer &);
std::ostream &operator<<(std::ostream &, const Consumers &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
