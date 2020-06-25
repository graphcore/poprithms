// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <sstream>
#include <string>

#include <poprithms/outline/linear/error.hpp>
#include <poprithms/outline/linear/linearusings.hpp>

namespace poprithms {
namespace outline {
namespace linear {

std::ostream &operator<<(std::ostream &ost, DType t) {
  switch (t) {
  case DType::INT32: {
    ost << std::string("INT32");
    break;
  }
  case DType::FLOAT32: {
    ost << std::string("FLOAT32");
    break;
  }
  case DType::FLOAT16: {
    ost << std::string("FLOAT16");
    break;
  }
  case DType::N:
    throw error("N is not a DType");
  }

  return ost;
}

} // namespace linear
} // namespace outline
} // namespace poprithms
