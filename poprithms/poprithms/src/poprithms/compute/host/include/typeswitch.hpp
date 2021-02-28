// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_TYPESWITCH_HPP
#define POPRITHMS_COMPUTE_HOST_TYPESWITCH_HPP

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>

#include <compute/host/include/basedata.hpp>
#include <compute/host/include/baseoperators.hpp>
#include <compute/host/include/ieeehalf.hpp>
#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/viewchange.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace compute {
namespace host {

/**
 * This template function can be used with any class with static method called
 * \a go.
 * \see TypedConcat_ for example.
 * */
template <typename F, class ReturnType, class... Args>
ReturnType typeSwitch(ndarray::DType t, Args &&... args) {
  switch (t) {
  case (ndarray::DType::Float64): {
    return F::template go<double>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Float32): {
    return F::template go<float>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Float16): {
    return F::template go<IeeeHalf>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Boolean): {
    return F::template go<bool>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Int8): {
    return F::template go<int8_t>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Unsigned8): {
    return F::template go<uint8_t>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Int16): {
    return F::template go<int16_t>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Unsigned16): {
    return F::template go<uint16_t>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Int32): {
    return F::template go<int32_t>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Unsigned32): {
    return F::template go<uint32_t>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Int64): {
    return F::template go<int64_t>(std::forward<Args>(args)...);
  }
  case (ndarray::DType::Unsigned64): {
    return F::template go<uint64_t>(std::forward<Args>(args)...);
  }
  case ndarray::DType::N:
  default: {
    throw error("invalid / unimplemented type in typeSwitch");
  }
  }
}

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
