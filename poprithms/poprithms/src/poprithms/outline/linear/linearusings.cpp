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
    return ost;
  }
  case DType::FLOAT32: {
    ost << std::string("FLOAT32");
    return ost;
  }
  case DType::FLOAT16: {
    ost << std::string("FLOAT16");
    return ost;
  }
  case DType::N:
  default:
    throw error("Invalid DType");
  }
}

} // namespace linear
} // namespace outline
} // namespace poprithms
