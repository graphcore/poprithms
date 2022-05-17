// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_CIRCULARCOUNTER_HPP
#define POPRITHMS_UTIL_CIRCULARCOUNTER_HPP

#include <ostream>
#include <sstream>
#include <unordered_map>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace util {

/**
 * An integer which can be incremented with modular arithmetic.
 * */
class CircularCounter {
public:
  CircularCounter(uint64_t m) : state_(0), modulus_(m) {
    if (modulus_ == 0) {
      throw poprithms::error::error("util", "Modulus cannot be 0.");
    }
  }
  void increment() { state_ = (state_ + 1) % modulus_; }

  /**
   * The current value of the integer (0 <= state < modulus).
   * */
  uint64_t state() const { return state_; }

  uint64_t getModulus() const { return modulus_; }

private:
  uint64_t state_;
  uint64_t modulus_;
};

/**
 * A map of CircularCounters, with templatized map key, Key.
 * */
template <class Key> class CircularCounters {
public:
  /**
   * Insert a CircularCounter with modulus #modulus_.
   * */
  void insert(Key key, uint64_t modulus_) {
    const auto found = counters.find(key);
    if (found != counters.cend()) {
      std::ostringstream oss;
      oss << "Failure in CircularCounters::insert for Key=" << key
          << " and modulus=" << modulus_
          << ". A CircularCounter already exists (with modulus="
          << found->second.getModulus() << ") for Key=" << key << ".";
      throw poprithms::error::error("util", oss.str());
    }
    counters.insert({key, modulus_});
  }

  /**
   * Increment the CircularCounter at Key #key.
   * */
  void increment(Key key) {
    auto found = counters.find(key);
    if (found == counters.cend()) {
      std::ostringstream oss;
      oss << "Invalid Key=" << key << '.';
      throw poprithms::error::error("util", oss.str());
    }
    found->second.increment();
  }

  uint64_t state(Key key) const {
    auto found = counters.find(key);
    if (found == counters.cend()) {
      std::ostringstream oss;
      oss << "Invalid Key=" << key << '.';
      throw poprithms::error::error("util", oss.str());
    }
    return found->second.state();
  }

private:
  std::unordered_map<Key, CircularCounter> counters;
};

} // namespace util
} // namespace poprithms

#endif
