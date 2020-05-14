#include <poprithms/outline/linear/error.hpp>

namespace poprithms {
namespace outline {
namespace linear {

poprithms::util::error error(const std::string &what) {
  static const std::string linear("outline::linear");
  return poprithms::util::error(linear, what);
}

} // namespace linear
} // namespace outline
} // namespace poprithms
