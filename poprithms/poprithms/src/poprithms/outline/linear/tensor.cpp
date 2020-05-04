#include <poprithms/outline/linear/tensor.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace outline {
namespace linear {

void Tensor::append(std::ostream &ost) const {
  ost << "debugStr:" << debugStr() << "  id:" << id() << "  type:" << type()
      << "  shape:(";
  poprithms::util::append(ost, shape());
  ost << ')';
}

std::ostream &operator<<(std::ostream &ost, const Tensor &t) {
  t.append(ost);
  return ost;
}

} // namespace linear
} // namespace outline
} // namespace poprithms
