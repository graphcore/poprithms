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

template class ViewChange<uint64_t>;
template class TypedData<uint64_t>;
template class ViewData<uint64_t>;
template class OriginData<uint64_t>;
template class AllocData<uint64_t>;
template class PointerData<uint64_t>;
template BaseDataSP
TypedConcat::go<uint64_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<uint64_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

template class ViewChange<int64_t>;
template class TypedData<int64_t>;
template class ViewData<int64_t>;
template class OriginData<int64_t>;
template class AllocData<int64_t>;
template class PointerData<int64_t>;
template BaseDataSP
TypedConcat::go<int64_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<int64_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

} // namespace host
} // namespace compute
} // namespace poprithms
