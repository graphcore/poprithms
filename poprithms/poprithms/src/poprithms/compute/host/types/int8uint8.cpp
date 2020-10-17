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

template class ViewChange<uint8_t>;
template class TypedData<uint8_t>;
template class ViewData<uint8_t>;
template class OriginData<uint8_t>;
template class AllocData<uint8_t>;
template class PointerData<uint8_t>;
template BaseDataSP
TypedConcat::go<uint8_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<uint8_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

template class ViewChange<int8_t>;
template class TypedData<int8_t>;
template class ViewData<int8_t>;
template class OriginData<int8_t>;
template class AllocData<int8_t>;
template class PointerData<int8_t>;
template BaseDataSP
TypedConcat::go<int8_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<int8_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

} // namespace host
} // namespace compute
} // namespace poprithms
