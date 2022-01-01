// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_TENSORMAP_HPP
#define POPRITHMS_COMMON_MULTIOUT_TENSORMAP_HPP

#include <string>
#include <vector>

#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/error/error.hpp>

namespace poprithms {
namespace common {
namespace multiout {

/**
 * A utility class to store data relating to Tensors in a multiout::Graph, but
 * not inside the Op classes. It is useful because it runs checks on the
 * validity of TensorIds when accessing data.
 * */
template <class Value> class TensorMap {

public:
  using Values = std::vector<Value>;

  /**
   * Initialize this TensorMap. The values in #v are assumed to be of the form
   * v[opId][outIndex], that is v[opId] contains all of the outputs of the Op
   * with OpId opId.
   * */
  TensorMap(std::vector<std::vector<Value>> &&v) : values(std::move(v)) {}
  TensorMap() = default;

  TensorMap &operator=(TensorMap &&) = default;
  TensorMap &operator=(const TensorMap &) = default;

  TensorMap(const TensorMap &) = default;
  TensorMap(TensorMap &&)      = default;

  virtual ~TensorMap() = default;

  void push_back(const std::vector<Value> &vs) { values.push_back(vs); }
  void push_back(std::vector<Value> &&vs) { values.push_back(std::move(vs)); }

  /**
   * Get the Value corresponding to the Tensor #tId.
   * */
  Value getValue(const TensorId &tId) const {
    assertValidTensorId(tId);
    return values[tId.opId().get()][tId.outIndex().get()];
  }

  /**
   * Get the Value (by non-const ref) corresponding to the Tensor #tId.
   * */
  Value &operator[](const TensorId &tId) {
    assertValidTensorId(tId);
    return values[tId.opId().get()][tId.outIndex().get()];
  }

  /**
   * Get the Values corresponding to all of the Tensors in #tIds.
   * */
  Values getValues(const TensorIds &tIds) const {
    Values vs;
    vs.reserve(tIds.size());
    for (auto tId : tIds) {
      vs.push_back(getValue(tId));
    }
    return vs;
  }

  /**
   * Set the Value corresponding to the Tensor #tId to #v.
   * */
  void setValue(const TensorId &tId, const Value &v) {
    assertValidTensorId(tId);
    values[tId.opId().get()][tId.outIndex().get()] = v;
  }

  void setValues(OpId opId, const Values &vs) {
    assertValidOpId(opId);
    values[opId.get()] = vs;
  }

  /**
   * Set the Value corresponding to the Tensor #tId to #v.
   * */
  void setValue(const TensorId &tId, Value &&v) {
    assertValidTensorId(tId);
    values[tId.opId().get()][tId.outIndex().get()] = std::move(v);
  }

private:
  void assertValidOpId(OpId opId) const {

    if (values.size() <= opId.get()) {
      throw poprithms::error::error(
          "common::multiout",
          "Invalid OpId, " + std::to_string(opId.get()) + ". Only " +
              std::to_string(values.size()) + " Ops in this TensorMap.");
    }
  }
  /**
   * Assert that the Tensor #tId has a Value stored for it.
   * */
  void assertValidTensorId(const TensorId &tId) const {
    uint64_t opId     = tId.opId().get();
    uint64_t outIndex = tId.outIndex().get();

    assertValidOpId(opId);

    if (values[opId].size() <= outIndex) {
      throw poprithms::error::error(
          "common::multiout",
          "Invalid TensorId, " + tId.str() + ". Only " +
              std::to_string(values[opId].size()) + " outputs for Op " +
              std::to_string(opId) + " in this TensorMap, so the OutIndex " +
              std::to_string(outIndex) + " is too large.");
    }
  }

  std::vector<std::vector<Value>> values;
};

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
