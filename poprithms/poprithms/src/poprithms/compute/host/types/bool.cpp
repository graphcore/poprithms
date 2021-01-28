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
