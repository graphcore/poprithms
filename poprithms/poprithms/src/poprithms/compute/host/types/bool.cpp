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

template class ViewChange<bool>;
template class TypedData<bool>;
template class ViewData<bool>;
template class OriginData<bool>;
template class AllocData<bool>;
template class PointerData<bool>;
template BaseDataSP
TypedConcat::go<bool>(const ConstDataPtrs &, const Shapes &, uint64_t);
template BaseDataSP
TypedConcat_::go<bool>(const ConstDataPtrs &, const Shapes &, uint64_t);

} // namespace host
} // namespace compute
} // namespace poprithms
