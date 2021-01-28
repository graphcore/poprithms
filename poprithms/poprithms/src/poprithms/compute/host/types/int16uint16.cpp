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

template class ViewChange<uint16_t>;
template class TypedData<uint16_t>;
template class ViewData<uint16_t>;
template class OriginData<uint16_t>;
template class AllocData<uint16_t>;
template class PointerData<uint16_t>;
template BaseDataSP
TypedConcat::go<uint16_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<uint16_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

template class ViewChange<int16_t>;
template class TypedData<int16_t>;
template class ViewData<int16_t>;
template class OriginData<int16_t>;
template class AllocData<int16_t>;
template class PointerData<int16_t>;
template BaseDataSP
TypedConcat::go<int16_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<int16_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

} // namespace host
} // namespace compute
} // namespace poprithms
