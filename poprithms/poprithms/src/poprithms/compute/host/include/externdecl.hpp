// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_EXTERNDECL_HPP
#define POPRITHMS_COMPUTE_HOST_EXTERNDECL_HPP

// Include this header to reduce the compile time of a translation unit. All
// of these templates are guaranteed to be instantiated elsewhere (see types
// directory).

#include <compute/host/include/allocdata.hpp>
#include <compute/host/include/basedata.hpp>
#include <compute/host/include/origindata.hpp>
#include <compute/host/include/pointerdata.hpp>
#include <compute/host/include/typedconcat.hpp>
#include <compute/host/include/typeddata.hpp>
#include <compute/host/include/viewdata.hpp>

#include <poprithms/compute/host/viewchange.hpp>

namespace poprithms {
namespace compute {
namespace host {

extern template class ViewChange<bool>;
extern template class TypedData<bool>;
extern template class ViewData<bool>;
extern template class OriginData<bool>;
extern template class AllocData<bool>;
extern template class PointerData<bool>;
extern template BaseDataSP
TypedConcat::go<bool>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<bool>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<uint8_t>;
extern template class TypedData<uint8_t>;
extern template class ViewData<uint8_t>;
extern template class OriginData<uint8_t>;
extern template class AllocData<uint8_t>;
extern template class PointerData<uint8_t>;
extern template BaseDataSP
TypedConcat::go<uint8_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<uint8_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<int8_t>;
extern template class TypedData<int8_t>;
extern template class ViewData<int8_t>;
extern template class OriginData<int8_t>;
extern template class AllocData<int8_t>;
extern template class PointerData<int8_t>;
extern template BaseDataSP
TypedConcat::go<int8_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<int8_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<uint16_t>;
extern template class TypedData<uint16_t>;
extern template class ViewData<uint16_t>;
extern template class OriginData<uint16_t>;
extern template class AllocData<uint16_t>;
extern template class PointerData<uint16_t>;
extern template BaseDataSP
TypedConcat::go<uint16_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<uint16_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<int16_t>;
extern template class TypedData<int16_t>;
extern template class ViewData<int16_t>;
extern template class OriginData<int16_t>;
extern template class AllocData<int16_t>;
extern template class PointerData<int16_t>;
extern template BaseDataSP
TypedConcat::go<int16_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<int16_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<uint32_t>;
extern template class TypedData<uint32_t>;
extern template class ViewData<uint32_t>;
extern template class OriginData<uint32_t>;
extern template class AllocData<uint32_t>;
extern template class PointerData<uint32_t>;
extern template BaseDataSP
TypedConcat::go<uint32_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<uint32_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<int32_t>;
extern template class TypedData<int32_t>;
extern template class ViewData<int32_t>;
extern template class OriginData<int32_t>;
extern template class AllocData<int32_t>;
extern template class PointerData<int32_t>;
extern template BaseDataSP
TypedConcat::go<int32_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<int32_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<uint64_t>;
extern template class TypedData<uint64_t>;
extern template class ViewData<uint64_t>;
extern template class OriginData<uint64_t>;
extern template class AllocData<uint64_t>;
extern template class PointerData<uint64_t>;
extern template BaseDataSP
TypedConcat::go<uint64_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<uint64_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<int64_t>;
extern template class TypedData<int64_t>;
extern template class ViewData<int64_t>;
extern template class OriginData<int64_t>;
extern template class AllocData<int64_t>;
extern template class PointerData<int64_t>;
extern template BaseDataSP
TypedConcat::go<int64_t>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<int64_t>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<IeeeHalf>;
extern template class TypedData<IeeeHalf>;
extern template class ViewData<IeeeHalf>;
extern template class OriginData<IeeeHalf>;
extern template class AllocData<IeeeHalf>;
extern template class PointerData<IeeeHalf>;
extern template BaseDataSP
TypedConcat::go<IeeeHalf>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<IeeeHalf>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<float>;
extern template class TypedData<float>;
extern template class ViewData<float>;
extern template class OriginData<float>;
extern template class AllocData<float>;
extern template class PointerData<float>;
extern template BaseDataSP
TypedConcat::go<float>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<float>(const ConstDataPtrs &, const Shapes &, uint64_t);

extern template class ViewChange<double>;
extern template class TypedData<double>;
extern template class ViewData<double>;
extern template class OriginData<double>;
extern template class AllocData<double>;
extern template class PointerData<double>;
extern template BaseDataSP
TypedConcat::go<double>(const ConstDataPtrs &, const Shapes &, uint64_t);
extern template BaseDataSP
TypedConcat_::go<double>(const ConstDataPtrs &, const Shapes &, uint64_t);

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
