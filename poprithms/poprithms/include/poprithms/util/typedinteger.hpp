// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_TYPEDINTEGER_HPP
#define POPRITHMS_UTIL_TYPEDINTEGER_HPP

#include <cstdint>
#include <functional>
#include <sstream>

namespace poprithms {
namespace util {

// Consider code:
//
// >  using OpId     = int;
// >  using TensorId = int;
// >  OpId a          = 11;
// >   .
// >  TensorId b      = a;
//
// This code suggests a bug : why is a TensorId being set to an OpId?
// TensorId's and OpId's should not be the same type.
//
// The following class gives C++'s "using" a form of typing, so
// that comparisons such as the 1 above will result in compiler errors.
//
// This code will find the presumably invalid copy at compile time:
//
// >  using OpId     = TypedInteger<'O', int>;
// >  using TensorId = TypedInteger<'T', int>;
// >  OpId a(11);
// >    .
// >    .
// >  TensorId b = a; // compilation error!
//
// Note that TensorId b(a.get()); will still work.
//

template <char T, typename INT> class TypedInteger {
public:
  TypedInteger()                      = default;
  TypedInteger(TypedInteger &&x)      = default;
  TypedInteger(const TypedInteger &x) = default;
  TypedInteger &operator=(TypedInteger &&x) = default;
  TypedInteger &operator=(const TypedInteger &x) = default;

  template <typename INT2> TypedInteger(INT2 v_) : v(static_cast<INT>(v_)) {
    static_assert(std::is_integral<INT2>::value,
                  "TypedIntegers can only be constructed from integer types");
  }

  INT get() const { return static_cast<INT>(v); }

  bool operator==(const TypedInteger &rhs) const {
    return get() == rhs.get();
  }
  bool operator<(const TypedInteger &rhs) const { return get() < rhs.get(); }
  bool operator<=(const TypedInteger &rhs) const {
    return get() <= rhs.get();
  }

  void operator+=(INT q) { v += q; }
  void operator-=(INT q) { v -= q; }

  void operator++() { v += 1; }
  void operator--() { v -= 1; }

  bool operator!=(const TypedInteger &rhs) const { return !operator==(rhs); }
  bool operator>(const TypedInteger &rhs) const { return !operator<=(rhs); }
  bool operator>=(const TypedInteger &rhs) const { return !operator<(rhs); }

private:
  INT v;
};

template <char T, typename INT>
std::ostream &operator<<(std::ostream &ost, TypedInteger<T, INT> id) {
  ost << id.get();
  return ost;
}

template <char T, typename INT>
std::string operator+(const std::string &s, TypedInteger<T, INT> id) {
  return s + std::to_string(id.get());
}

} // namespace util
} // namespace poprithms

// To enable hashing of new classes, this is the recommended approach from
// https://en.cppreference.com/w/cpp/utility/hash
// With this, it is possible to create std::unordered_sets of TypedIntegers.
namespace std {
template <char T, typename INT>
struct hash<poprithms::util::TypedInteger<T, INT>> {
  std::size_t operator()(poprithms::util::TypedInteger<T, INT> const &s) const
      noexcept {
    return std::hash<INT>{}(s.get());
  }
};
} // namespace std

#endif
