// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/viewchange.hpp>

namespace poprithms {
namespace compute {
namespace host {

Shape ViewChangeHelper::prePadToRank(const Shape &a, uint64_t r) {
  std::vector<int64_t> a_ = a.get();
  std::vector<int64_t> x(r - a.rank_u64(), 1);
  x.insert(x.end(), a_.cbegin(), a_.cend());
  return Shape(x);
}

void ViewChangeHelper::assertConcatSizes(uint64_t nPtrs, uint64_t nShapes) {
  if (nPtrs != nShapes) {
    std::ostringstream oss;
    oss << "Failure in asserConcatSizes(nPtrs=" << nPtrs
        << ", nShapes=" << nShapes << ").";
    throw error(oss.str());
  }
}

void ViewChangeHelper::assertExpandedNElms(uint64_t observed,
                                           uint64_t expected) {
  if (observed != expected) {
    std::ostringstream oss;
    oss << "Failure in assertExpandedNElms(observed=" << observed
        << ", expected=" << expected << ").";
    throw error(oss.str());
  }
}

void ViewChangeHelper::assertExpandableTo(const Shape &from,
                                          const Shape &to) {
  if (from.numpyBinary(to) != to) {
    std::ostringstream oss;
    oss << "Failure in assertExpandableTo(from=" << from << ", to=" << to
        << "). from.numpyBinary(to) is " << from.numpyBinary(to) << ", not "
        << to << ". to=" << to << " does not \"dominate\" from=" << from
        << '.';
    throw error(oss.str());
  }
}

std::vector<ViewChangeHelper::OldNew>
ViewChangeHelper::getTiled(const Shape &s, const Permutation &p) {

  // Use a tiling which has length 16 in all directions. The motivation for
  // this comes from the PopART implementation for a 2-D transpose. In
  // general, you can probably do better TODO(T27198).
  constexpr uint64_t tileLength{16};

  const auto indexToOld = s.getRowMajorBlockOrdered(tileLength);
  const auto newToOld   = s.getDimShuffledRowMajorIndices(p);

  std::vector<OldNew> oldAndNew(s.nelms_u64());
  std::transform(
      indexToOld.cbegin(),
      indexToOld.cend(),
      oldAndNew.begin(),
      [&newToOld](uint64_t n) {
        const auto o = newToOld[static_cast<uint64_t>(n)];
        return OldNew{static_cast<uint64_t>(o), static_cast<uint64_t>(n)};
      });

  return oldAndNew;
}

[[noreturn]] void ViewChangeHelper::nullptrDataNotAllowed() {
  throw error(
      "nullptr is not allowed in the constructor of ViewChange::Data");
}

// explicitly instantiate the ViewChange template class for the types promised
// in viewchange.hpp
template class ViewChange<int8_t>;
template class ViewChange<uint8_t>;
template class ViewChange<int16_t>;
template class ViewChange<uint16_t>;
template class ViewChange<int32_t>;
template class ViewChange<uint32_t>;
template class ViewChange<int64_t>;
template class ViewChange<uint64_t>;
template class ViewChange<float>;
template class ViewChange<double>;

} // namespace host
} // namespace compute
} // namespace poprithms
