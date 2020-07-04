// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_NEST_OPTIONALSET_HPP
#define POPRITHMS_MEMORY_NEST_OPTIONALSET_HPP

#include <array>
#include <memory>

namespace poprithms {
namespace memory {
namespace nest {

/**
 * Optional Ts, similar in purpose to std::optional
 * N : the number of objects to store (all or none)
 * T : the type of the objects to store
 * */

template <uint64_t N, class T> class OptionalSet {
public:
  OptionalSet() = default;
  /**
   * Constructor for a full OptionalSet
   * */
  OptionalSet(std::array<T, N> x)
      : ts(std::unique_ptr<std::array<T, N>>(new std::array<T, N>(x))),
        contains(true) {}
  /**
   * Factory function for an empty OptionalSet
   * */
  static OptionalSet<N, T> None() { return OptionalSet<N, T>(); }

  /**
   * If this OptionalSet is full, return the i'th element.
   * If this OptionalSet is empty, undefined behaviour.
   * */
  template <uint64_t i> const T &get() const { return std::get<i>(*ts); }

  const T &first() const { return get<0>(); }

  bool empty() const { return !full(); }

  bool full() const { return contains; }

private:
  // T may not have a default constructor (it does not for the particular case
  // of a Sett), and so we allocate memory to avoid a complicated
  // implementation with bit arrays
  std::unique_ptr<std::array<T, N>> ts{nullptr};
  bool contains{false};
};

} // namespace nest
} // namespace memory
} // namespace poprithms

#endif
