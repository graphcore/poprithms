// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_TOPTIONALTENSOR_HPP
#define POPRITHMS_COMMON_MULTIOUT_TOPTIONALTENSOR_HPP

#include <ostream>

#include <poprithms/common/multiout/optionaltensorid.hpp>

namespace poprithms {
namespace common {
namespace multiout {

using multiout::OptionalTensorId;
using multiout::OptionalTensorIds;

/**
 * Template class for an optional tensor.
 *
 * \param T a tensor class, which has method id() which returns a TensorId.
 * */
template <typename T> class TOptionalTensor {

  using TOptionalTensors = std::vector<TOptionalTensor<T>>;

public:
  /**
   * Create an unset optional tensor.
   * */
  TOptionalTensor() = default;

  ~TOptionalTensor()                                  = default;
  TOptionalTensor(const TOptionalTensor &)            = default;
  TOptionalTensor(TOptionalTensor &&)                 = default;
  TOptionalTensor &operator=(const TOptionalTensor &) = default;
  TOptionalTensor &operator=(TOptionalTensor &&)      = default;

  /**
   * Construct an optional tensor from tensor #tensor.
   * */
  TOptionalTensor(const T &tensor) : t(tensor) {}
  TOptionalTensor(T &&tensor) : t(std::move(tensor)) {}

  /**
   * Obtain a vector of OptionalTensorIds from a vector of optional tensors.
   * */
  static OptionalTensorIds fromOptionalTensors(const TOptionalTensors ots) {
    OptionalTensorIds otIds;
    otIds.reserve(ots.size());
    for (const auto &ot : ots) {
      if (ot.has_value()) {
        otIds.push_back(ot.value().id());
      } else {
        // unset tensor id gets mapped to unset tensor.
        otIds.push_back({});
      }
    }
    return otIds;
  }

  template <typename CT> static TOptionalTensors fromTensors(const CT &ts) {
    TOptionalTensors opts;
    opts.reserve(ts.size());
    for (const auto &t : ts) {
      opts.push_back(TOptionalTensor(t));
    }
    return opts;
  }

  /**
   * Implicit casting from optional tensor to optional tensor id.
   * */
  operator OptionalTensorId() const {
    return has_value() ? value().id() : OptionalTensorId();
  }

  /**
   * \return The tensor, error if it is unset.
   * */
  const T &value() const {
    if (!has_value()) {
      throw poprithms::error::error(
          "common::multiout",
          "Invalid call to OptionalTensor::value(). has_value() is false.");
    }
    return t;
  }

  bool has_value() const { return t.graphIsSet(); }

  void append(std::ostream &ost) const {
    if (has_value()) {
      ost << value().id();
    } else {
      ost << "none";
    }
  }

private:
  T t{{0, 0}, nullptr};
};

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
