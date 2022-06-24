// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_TYPEDINTEGER_HPP
#define POPRITHMS_UTIL_TYPEDINTEGER_HPP

#include <cstdint>
#include <functional>
#include <sstream>
#include <type_traits>

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

class TypedIntegerBase {};

template <char T, typename INT> class TypedInteger : public TypedIntegerBase {
public:
  TypedInteger()                                 = default;
  TypedInteger(TypedInteger &&x)                 = default;
  TypedInteger(const TypedInteger &x)            = default;
  TypedInteger &operator=(TypedInteger &&x)      = default;
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

/**
 *
 * When implementing template methods with a template parameter Q which is
 * EITHER a TypedIntegers OR a C++ integral types, we sometimes need get a
 * uint64_t from Q. For C++ integral types this just involves a
 * static_cast<uint64_t>(q), for TypedIntegers it requires calling the get()
 * method first  (static_cast<uint64_t>(q.get()). We don't want to implement
 * an implicit cast for TypedIntegers (unint64_t operator()) because the
 * purpose of that class is to get compiler errors when types are incorrectly
 * used.
 *
 * This helper class removes this indirection, so that for both TypedIntegers
 * and C++ integral types, a call to IntValGetter<Q>::get_u64(q) returns the
 * desired uint64_t.
 * */
template <typename Q, class Enable = void> class IntValGetter {};

/**
 * The typed integer case:
 * */
template <typename Q>
class IntValGetter<
    Q,
    typename std::enable_if<
        std::is_base_of<poprithms::util::TypedIntegerBase, Q>::value>::type> {
public:
  static uint64_t get_u64(Q q) { return static_cast<uint64_t>(q.get()); }
};

/**
 * The C++ native integer case:
 * */
template <typename Q>
class IntValGetter<
    Q,
    typename std::enable_if<std::is_integral<Q>::value>::type> {
public:
  static uint64_t get_u64(Q q) { return static_cast<uint64_t>(q); }
};

} // namespace util
} // namespace poprithms

// To enable hashing of new classes, this is the recommended approach from
// https://en.cppreference.com/w/cpp/utility/hash
// With this, it is possible to create std::unordered_sets of TypedIntegers.
namespace std {
template <char T, typename INT>
struct hash<poprithms::util::TypedInteger<T, INT>> {
  std::size_t
  operator()(poprithms::util::TypedInteger<T, INT> const &s) const noexcept {
    return std::hash<INT>{}(s.get());
  }
};

} // namespace std

#endif
