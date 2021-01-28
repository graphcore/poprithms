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

template class ViewChange<uint32_t>;
template class TypedData<uint32_t>;
template class ViewData<uint32_t>;
template class OriginData<uint32_t>;
template class AllocData<uint32_t>;
template class PointerData<uint32_t>;
template BaseDataSP
TypedConcat::go<uint32_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<uint32_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

template class ViewChange<int32_t>;
template class TypedData<int32_t>;
template class ViewData<int32_t>;
template class OriginData<int32_t>;
template class AllocData<int32_t>;
template class PointerData<int32_t>;
template BaseDataSP
TypedConcat::go<int32_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<int32_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

} // namespace host
} // namespace compute
} // namespace poprithms
