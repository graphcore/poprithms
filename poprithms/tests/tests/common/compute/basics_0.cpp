#include <poprithms/common/compute/initialvalues.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::common::compute;
void testInitialValues0() {

  InitialValues inVals(2);
  inVals.setValue(OutIndex(0), 3, HostTensor::float32(17));

  auto inVals2 = inVals;

  if (inVals2 != inVals) {
    throw poprithms::test::error("Comparison of copied InitialValues failed");
  }

  InitialValues inVals3(2);
  inVals3.setValue(OutIndex(0), 3, HostTensor::float32(17));

  if (inVals3 != inVals) {
    throw poprithms::test::error(
        "Comparison of numerically equivalent InitialValues failed");
  }

  InitialValues inVals4(2);
  inVals4.setValue(OutIndex(0), 3, HostTensor::float32(17.001));

  if (inVals4 == inVals) {
    throw poprithms::test::error(
        "Comparison of numerically different InitialValues failed");
  }

  InitialValues inVals5(2);
  inVals5.setValue(OutIndex(0), 3, inVals.getInitialValues(0).at(3).copy());
  if (inVals5 != inVals) {
    throw poprithms::test::error(
        "Comparison of numerically equivalent InitialValues failed, value "
        "obtained my introspection");
  }
}
} // namespace

int main() {

  testInitialValues0();

  return 0;
}
