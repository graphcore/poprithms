// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../include/allocdata.hpp"
#include "../include/basedata.hpp"
#include "../include/externdecl.hpp"
#include "../include/pointerdata.hpp"
#include "../include/typedconcat.hpp"
#include "../include/viewdata.hpp"

// The Float16 type has a few additional template methods to specialize:
namespace poprithms {
namespace ndarray {
template <> DType get<copied_from_poplar::IeeeHalf>() {
  return DType::Float16;
}
} // namespace ndarray
} // namespace poprithms
namespace poprithms {
namespace compute {
namespace host {
template <>
std::vector<IeeeHalf> castPtrToVector<>(const double *from, uint64_t nElms) {
  auto a = castPtrToVector<double, float>(from, nElms);
  return castPtrToVector<float, IeeeHalf>(a.data(), nElms);
}
} // namespace host
} // namespace compute
} // namespace poprithms

// The standard templates, being instantiated:
namespace poprithms {
namespace compute {
namespace host {
template class ViewChange<IeeeHalf>;
template class TypedData<IeeeHalf>;
template class ViewData<IeeeHalf>;
template class OriginData<IeeeHalf>;
template class AllocData<IeeeHalf>;
template class PointerData<IeeeHalf>;
template BaseDataSP
TypedConcat::go<IeeeHalf>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<IeeeHalf>(const ConstDataPtrs &, const Shapes &, uint64_t);

} // namespace host
} // namespace compute
} // namespace poprithms
