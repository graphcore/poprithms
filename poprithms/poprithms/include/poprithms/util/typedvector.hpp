// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_POPRITHMS_UTIL_TYPEDVECTOR_HPP
#define GUARD_POPRITHMS_UTIL_TYPEDVECTOR_HPP

#include <sstream>
#include <vector>

#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace util {

/** A thin wrapper around a std::vector to make it more strongly typed. This
 * is useful to allow the compiler to detect errors where semantically
 * different vectors are used incorrectly. For example, consider a function
 *
 * Tensor
 * slice(const std::vector<int> & dimensions, const std::vector<int> & sizes);
 *
 * It is easy for a user to call slice(mySizes, myDimensions), that is to
 * accidentally switch the 2 semantically different vectors, without getting
 * a compilation error.
 *
 * If instead the function signature looked like:
 *
 * using Dimensions = TypedVector<int, 'D', 'I', 'M', 'S'>;
 * using Sizes = TypedVector<int, 'S', 'I', 'Z', 'E', 'S'>;
 * Tensor slice(const Dimensions &, const Sizes &);
 *
 * Then the user's error would be caught at compile time.
 *
 * Note that this wrapping has no effect on runtime performance.
 *
 * \sa util::TypedInteger, a similar class for integer types.
 * */
template <typename INT, char... TypeParams> class TypedVector {
public:
  TypedVector() = default;
  TypedVector(const std::vector<INT> &vals__) : vals_(vals__) {}
  TypedVector(std::initializer_list<INT> init) : vals_(init) {}

  /** A selection of wrapped comparison operators */
  bool operator==(const TypedVector &rhs) const { return get() == rhs.get(); }
  bool operator!=(const TypedVector &rhs) const { return get() != rhs.get(); }

  /** A selection of other interfaces */
  size_t size() const { return get().size(); }
  INT &operator[](size_t i) { return vals_[i]; }
  const INT &operator[](size_t i) const { return vals_[i]; }

  /**
   * To access the other members of the underlying vector (size, back, begin,
   * etc) the user must go via these getters:
   * */
  const std::vector<INT> &get() const { return vals_; }
  std::vector<INT> &get() { return vals_; }

private:
  std::vector<INT> vals_;
};

template <typename INT, char... TypeParams>
std::ostream &operator<<(std::ostream &ost,
                         const TypedVector<INT, TypeParams...> &v) {
  poprithms::util::append(ost, v.get());
  return ost;
}

} // namespace util
} // namespace poprithms

#endif
