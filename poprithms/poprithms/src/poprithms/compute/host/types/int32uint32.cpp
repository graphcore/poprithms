// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../include/allocdata.hpp"
#include "../include/basedata.hpp"
#include "../include/externdecl.hpp"
#include "../include/pointerdata.hpp"
#include "../include/typedconcat.hpp"
#include "../include/viewdata.hpp"

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
