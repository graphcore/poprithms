// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_ROWMAJORALLOC_HPP
#define POPRITHMS_COMPUTE_HOST_ROWMAJORALLOC_HPP

#include <system_error>

#include <boost/container/vector.hpp>

#include <compute/host/include/origindata.hpp>

namespace poprithms {
namespace compute {
namespace host {

class Serializer;

// We do not want to use std::vector<bool> because:
//
//  1) does not expose raw data through a data() member method.
//  2) parallelization requires mutex's as different indices in the vector
//     correspond to the same byte address.
//
// To avoid the class, we use the boost vector class for bools. The boost
// vector class is guaranteed to not be specialized like the std::vector is.

// The default (non-bool) case:
template <typename T> class Container {
public:
  using Primal = std::vector<T>;
  using Dual   = boost::container::vector<bool>;
};

// Special case handling for bools:
template <> class Container<bool> {
public:
  // Data stored in this format:
  using Primal = boost::container::vector<bool>;

  // Data can be constructed from this format:
  using Dual = std::vector<bool>;
};

// Convert from one vector type to another
template <typename VFrom, typename VTo> VTo convertVector(const VFrom &from) {
  VTo to;
  to.reserve(from.size());
  to.insert(to.end(), from.cbegin(), from.cend());
  return to;
}

/**
 * A data-storing OriginData.
 * */
template <class T> class AllocData : public OriginData<T> {
private:
  using Primal = typename Container<T>::Primal;
  using Dual   = typename Container<T>::Dual;
  friend class Serializer;

public:
  AllocData(const Primal &v) : up(std::make_unique<Primal>(v)) {}
  AllocData(Primal &&v) : up(std::make_unique<Primal>(std::move(v))) {}

  AllocData(const Dual &v)
      : up(std::make_unique<Primal>(convertVector<Dual, Primal>(v))) {}
  AllocData(Dual &&v) : AllocData(convertVector<Dual, Primal>(v)) {}

  AllocData(T f) : AllocData(Primal{f}) {}

  void append(std::ostream &ost) const final {
    ost << "AllocData(dtype=" << poprithms::ndarray::lcase<T>()
        << ",nelms=" << nelms_u64() << ')';
  }

  T *dataPtr() const final { return up->data(); }

  uint64_t nelms_u64() const final { return up->size(); }

  BaseDataSP clone() const final {
    auto x = *up;
    return std::make_shared<AllocData<T>>(std::move(x));
  }

private:
  std::unique_ptr<Primal> up;
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
