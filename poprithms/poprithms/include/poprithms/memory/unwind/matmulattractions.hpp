// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_MATMULATTRACTIONS_HPP
#define POPRITHMS_MEMORY_UNWIND_MATMULATTRACTIONS_HPP

#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#include <poprithms/common/multiout/tensorid.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using common::multiout::InIndex;
using common::multiout::OpId;
using common::multiout::TensorId;
class MatMulAttractions {

public:
  static MatMulAttractions Default() { return MatMulAttractions{}; }

  /**
   * The importance of having the left-hand side input have a
   * specialized layout (like that returned by poplibs' createMatMulInputLhs).
   * Specifically, how many points are obtained per element if the layouts
   * match.
   * */
  double lhs() const { return lhs_; }

  /**
   * The importance of having the left-hand side input have a
   * specialized layout (like that returned by poplibs' createMatMulInputLhs).
   * */
  double rhs() const { return rhs_; }

  /**
   * The importance of having the left-hand side input have the same layout as
   * the matmul output.
   * */
  double lhsOut() const { return lhsOut_; }

  /**
   * The importance of having the right-hand side input have the same layout
   * as the matmul output.
   * */
  double rhsOut() const { return rhsOut_; }

  MatMulAttractions &lhs(double d) {
    lhs_ = d;
    return *this;
  }

  MatMulAttractions &rhs(double d) {
    rhs_ = d;
    return *this;
  }

  MatMulAttractions &lhsOut(double d) {
    lhsOut_ = d;
    return *this;
  }

  MatMulAttractions &rhsOut(double d) {
    rhsOut_ = d;
    return *this;
  }

  bool operator==(const MatMulAttractions &rhs) const {
    return tup() == rhs.tup();
  }
  bool operator!=(const MatMulAttractions &rhs) const {
    return !operator==(rhs);
  }

  std::tuple<double, double, double, double> tup() const {
    return std::tuple<double, double, double, double>{
        lhs_, rhs_, lhsOut_, rhsOut_};
  }

private:
  double lhs_    = 100.;
  double rhs_    = 100.;
  double lhsOut_ = 50.;
  double rhsOut_ = 50.;

  MatMulAttractions() = default;
};

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
