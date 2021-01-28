// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <compute/host/include/allocdata.hpp>
#include <compute/host/include/basedata.hpp>
#include <compute/host/include/externdecl.hpp>
#include <compute/host/include/pointerdata.hpp>
#include <compute/host/include/typedconcat.hpp>
#include <compute/host/include/viewdata.hpp>

namespace poprithms {
namespace compute {
namespace host {

template class ViewChange<float>;
template class TypedData<float>;
template class ViewData<float>;
template class OriginData<float>;
template class AllocData<float>;
template class PointerData<float>;
template BaseDataSP
TypedConcat::go<float>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<float>(const ConstDataPtrs &, const Shapes &, uint64_t);

template class ViewChange<double>;
template class TypedData<double>;
template class ViewData<double>;
template class OriginData<double>;
template class AllocData<double>;
template class PointerData<double>;
template BaseDataSP
TypedConcat::go<double>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<double>(const ConstDataPtrs &, const Shapes &, uint64_t);

} // namespace host
} // namespace compute
} // namespace poprithms
