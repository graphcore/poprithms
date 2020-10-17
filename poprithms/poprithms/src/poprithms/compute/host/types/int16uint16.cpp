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
