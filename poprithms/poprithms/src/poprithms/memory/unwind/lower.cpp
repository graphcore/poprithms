// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <memory/unwind/error.hpp>

#include <poprithms/memory/unwind/lower.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

void loweringShapeAssert(const Shape &a,
                         const Shape &b,
                         const std::string &ctxt) {
  if (a != b) {
    std::ostringstream oss;
    oss << "Failed in loweringShapeAssert with Shapes " << a << " and " << b
        << ". The context provided is: " << ctxt;
    throw error(oss.str());
  }
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
