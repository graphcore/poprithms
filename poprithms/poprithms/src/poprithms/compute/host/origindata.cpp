// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "./include/basedata.hpp"
#include "./include/externdecl.hpp"

namespace poprithms {
namespace compute {
namespace host {

std::vector<int64_t> OriginDataHelper::getIota_i64(uint64_t N) {
  std::vector<int64_t> iotic(N);
  std::iota(iotic.begin(), iotic.end(), 0);
  return iotic;
}

void OriginDataHelper::assertSameBinaryOpNelms(uint64_t n0,
                                               uint64_t n1,
                                               const BaseData &td) {
  if (n0 != n1) {
    std::ostringstream oss;
    oss << "Failure in assertSameBinaryOpNelms: " << n0 << " != " << n1
        << " for " << td;
    throw error(oss.str());
  }
}

std::vector<uint16_t> OriginDataHelper::float16ToUint16(
    const std::vector<IeeeHalf> &asIeeeFloat16) {
  const auto N = asIeeeFloat16.size();
  std::vector<uint16_t> asUint16s;
  asUint16s.reserve(N);
  for (uint64_t i = 0; i < N; ++i) {
    asUint16s.push_back(asIeeeFloat16[i].bit16());
  }
  return asUint16s;
}

} // namespace host
} // namespace compute
} // namespace poprithms
