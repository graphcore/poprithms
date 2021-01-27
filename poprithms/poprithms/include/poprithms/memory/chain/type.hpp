// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_CHAIN_TYPE_HPP
#define POPRITHMS_MEMORY_CHAIN_TYPE_HPP

#include <ostream>

namespace poprithms {
namespace memory {
namespace chain {
/** Unlike most view-changing Graph projects in poprithms, the Chain project
 * does not use polymorphism for the different Op types. This is to facilitate
 * the template method "get" in the Graph class, used by the public template
 * method "apply".
 *
 * Instead of using polymorphism, each Op has an enum to describe how it
 * changes the view of a Tensor.
 * */
enum class Type {
  DimShuffle = 0,
  Expand,
  Reduce,
  Reshape,
  Reverse,
  SettSample,
  SettFillInto,
};

std::ostream &operator<<(std::ostream &, Type);
std::string getTypeString(Type);

} // namespace chain
} // namespace memory
} // namespace poprithms

#endif
