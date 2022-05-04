// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_DEVICEID_HPP
#define POPRITHMS_NDARRAY_DEVICEID_HPP

#include <string>
#include <vector>

namespace poprithms {
namespace ndarray {

/**
 * The id of a device. This class does not specify what a 'device' is, it is
 * really just a strongly typed integer.
 * */
class DeviceId {
public:
  DeviceId() = delete;
  DeviceId(uint32_t v) : val(v) {}
  bool operator==(const DeviceId &rhs) const { return val == rhs.val; }
  bool operator!=(const DeviceId &rhs) const { return !operator==(rhs); }
  bool operator<(const DeviceId &rhs) const { return val < rhs.val; }
  bool operator>(const DeviceId &rhs) const { return val > rhs.val; }
  bool operator<=(const DeviceId &rhs) const { return val <= rhs.val; }
  bool operator>=(const DeviceId &rhs) const { return val >= rhs.val; }
  std::string str() const { return std::to_string(val); }
  uint32_t get_u32() const { return val; }
  int64_t get_i64() const { return static_cast<int64_t>(val); }
  uint64_t get_u64() const { return static_cast<uint64_t>(val); }

private:
  uint32_t val;
};

using DeviceIds = std::vector<DeviceId>;
std::ostream &operator<<(std::ostream &, const DeviceIds &);
std::ostream &operator<<(std::ostream &, const DeviceId &);

} // namespace ndarray
} // namespace poprithms

namespace std {
template <> struct hash<poprithms::ndarray::DeviceId> {
  std::size_t
  operator()(const poprithms::ndarray::DeviceId &s) const noexcept {
    return std::hash<uint32_t>{}(s.get_u32());
  }
};
} // namespace std

#endif
