// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_POINTERDATA_HPP
#define POPRITHMS_COMPUTE_HOST_POINTERDATA_HPP

#include <compute/host/include/basedata.hpp>

namespace poprithms {
namespace compute {
namespace host {

/**
 * An OriginData which does not contain an internal buffer, only a raw
 * pointer to underlying data is kept.
 * */
template <class T> class PointerData : public OriginData<T> {
public:
  PointerData(T *data__, uint64_t nElms__) : data_(data__), nElms_(nElms__) {}

  void append(std::ostream &ost) const final {
    ost << "PointerData(dtype=" << poprithms::ndarray::lcase<T>()
        << ",nelms=" << nelms_u64() << ')';
  }

  T *dataPtr() const final { return data_; }

  uint64_t nelms_u64() const final { return nElms_; }

  BaseDataSP clone() const final {
    return std::make_shared<PointerData<T>>(data_, nElms_);
  }

  void updateData(T *n) { data_ = n; }

private:
  T *data_;
  const uint64_t nElms_;
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
