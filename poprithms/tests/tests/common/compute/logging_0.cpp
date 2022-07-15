// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iterator>
#include <numeric>

#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/error/error.hpp>

namespace {

using namespace poprithms::common::compute;

} // namespace

void checkAppearInScheduledOrder0() {

  SlickGraph g;
  auto mg         = g.createSubGraph("main");
  const auto host = g.host();
  auto a          = mg.variable(DType::Float32, {}, host);
  auto b          = mg.variable(DType::Float32, {}, host);
  auto foo        = a + b;
  auto bar        = a * b;
  g.constraint(bar.opId(), foo.opId());
  auto out = foo * bar;
  (void)out;

  // clang-format off
  std::vector<std::string> lines = {
     "OpId  OpType   InTensors        Shape  Graph       NonDataIns  Device      Type",
     "----  ------   ---------        -----  -----       ----------  ------      ----",
     "0     VarInit  ()               ()     main(id=0)  ()          Host(id=0)  float32",
     "1     VarInit  ()               ()     main(id=0)  ()          Host(id=0)  float32",
     "3     Mul      ((op=0),(op=1))  ()     main(id=0)  ()          Host(id=0)  float32",
     "2     Add      ((op=0),(op=1))  ()     main(id=0)  (3)         Host(id=0)  float32",
     "4     Mul      ((op=2),(op=3))  ()     main(id=0)  ()          Host(id=0)  float32"};
  // clang-format on

  std::ostringstream oss;
  oss << g;
  auto outString = oss.str();
  for (const auto &l : lines) {
    if (outString.find(l) == std::string::npos) {
      std::ostringstream errm;
      errm << "Did not find the string \n"
           << l << " \nin the logging string \n"
           << outString
           << "\nThis may not be a problem, this test is too brittle. Make a "
              "call, improve the test. ";
      throw poprithms::test::error(errm.str());
    }
  }

  // check that mul appears before add in summary
  auto found0 = outString.find("\n3   ");
  auto found1 = outString.find("\n2   ");
  if (found0 > found1) {
    throw poprithms::test::error("Expected op 3 to appear before op 2 in the "
                                 "summary because of explicit topo-con");
  }
}

int main() { checkAppearInScheduledOrder0(); }
