// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_MATMULATTRACTIONS_HPP
#define POPRITHMS_MEMORY_UNWIND_MATMULATTRACTIONS_HPP

#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/unwind/graph.hpp>

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

class MatMulSources {

public:
  MatMulSources(const TensorId &lhsSource,
                const TensorId &rhsSource,
                const TensorId &outSource)
      : lhsSource_(lhsSource), rhsSource_(rhsSource), outSource_(outSource) {}

  TensorId lhsSource() const { return lhsSource_; }
  TensorId rhsSource() const { return rhsSource_; }
  TensorId outSource() const { return outSource_; }

private:
  TensorId lhsSource_;
  TensorId rhsSource_;
  TensorId outSource_;
};

template <class TensorCreatorInserter>
MatMulSources growMatmul(const TensorCreatorInserter &tcInserter,
                         poprithms::memory::unwind::Graph &g,
                         const MatMulAttractions &atts,
                         const TensorId &lhs,
                         const TensorId &rhs) {

  auto name = [&tcInserter](const std::string &pre) -> std::string {
    return pre + "_matmul_source_" + std::to_string(tcInserter.opId().get());
  };

  const auto lhsShape = g.shape(lhs);
  const auto rhsShape = g.shape(rhs);
  const auto outShape = lhsShape.matmul(rhsShape);

  TensorId lhsSource{g.barrier({}, {lhsShape}, name("lhs")), 0};
  tcInserter.insertMatMulLhsCreator(lhsSource);
  g.insertValuedPair(lhs, lhsSource, atts.lhs());

  TensorId rhsSource{g.barrier({}, {rhsShape}, name("rhs")), 0};
  tcInserter.insertMatMulRhsCreator(rhsSource);
  g.insertValuedPair(rhs, rhsSource, atts.rhs());

  TensorId outSource{g.barrier({}, {outShape}, name("mm_out")), 0};
  tcInserter.insertMatMulOutCreator(outSource);

  if (lhsShape == outShape) {
    g.insertValuedPair(lhs, outSource, atts.lhsOut());
  }

  if (rhsShape == outShape) {
    g.insertValuedPair(rhs, outSource, atts.rhsOut());
  }

  return {lhsSource, rhsSource, outSource};
}

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
