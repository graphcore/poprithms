// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_TYPEDTENSORDATA_HPP
#define POPRITHMS_COMPUTE_HOST_TYPEDTENSORDATA_HPP

#include <memory>

#include <compute/host/include/basedata.hpp>
#include <compute/host/include/numpyformatter.hpp>

namespace poprithms {
namespace compute {
namespace host {

template <typename T> class TypedData : public BaseData {

public:
  ndarray::DType dtype() const final { return ndarray::get<T>(); }

  virtual std::vector<T> getNativeVector() const = 0;

  void appendValues(std::ostream &ost, const Shape &sh) const final {
    auto nvt = getNativeVector();
    std::vector<std::string> nv;
    nv.reserve(nvt.size());
    for (auto x : nvt) {
      nv.push_back(std::to_string(x));
    }
    NumpyFormatter::append(nv, ost, sh, 50);
  }

  std::shared_ptr<AllocData<float>> toFloat32() const final {
    return std::make_shared<AllocData<float>>(getFloat32Vector());
  }

  std::shared_ptr<AllocData<IeeeHalf>> toFloat16() const final {
    const auto dt = getFloat16Vector_u16();
    std::vector<IeeeHalf> vals;
    vals.reserve(dt.size());
    for (auto x : dt) {
      vals.push_back(IeeeHalf::fromBits(x));
    }
    return std::make_shared<AllocData<IeeeHalf>>(std::move(vals));
  }

  std::shared_ptr<AllocData<double>> toFloat64() const final {
    return std::make_shared<AllocData<double>>(getFloat64Vector());
  }

  std::shared_ptr<AllocData<int64_t>> toInt64() const final {
    return std::make_shared<AllocData<int64_t>>(getInt64Vector());
  }

  std::shared_ptr<AllocData<uint64_t>> toUnsigned64() const final {
    return std::make_shared<AllocData<uint64_t>>(getUnsigned64Vector());
  }

  std::shared_ptr<AllocData<int32_t>> toInt32() const final {
    return std::make_shared<AllocData<int32_t>>(getInt32Vector());
  }

  std::shared_ptr<AllocData<uint32_t>> toUnsigned32() const final {
    return std::make_shared<AllocData<uint32_t>>(getUnsigned32Vector());
  }

  std::shared_ptr<AllocData<int16_t>> toInt16() const final {
    return std::make_shared<AllocData<int16_t>>(getInt16Vector());
  }

  std::shared_ptr<AllocData<uint16_t>> toUnsigned16() const final {
    return std::make_shared<AllocData<uint16_t>>(getUnsigned16Vector());
  }

  std::shared_ptr<AllocData<int8_t>> toInt8() const final {
    return std::make_shared<AllocData<int8_t>>(getInt8Vector());
  }

  std::shared_ptr<AllocData<uint8_t>> toUnsigned8() const final {
    return std::make_shared<AllocData<uint8_t>>(getUnsigned8Vector());
  }

  std::shared_ptr<AllocData<bool>> toBool() const final {
    return std::make_shared<AllocData<bool>>(getBoolVector());
  }
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
