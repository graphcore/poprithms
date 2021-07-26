// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_WHERE_HPP
#define POPRITHMS_UTIL_WHERE_HPP

#include <algorithm>
#include <tuple>
#include <vector>

namespace poprithms {
namespace util {

/**
 * \return a vector #mask of the same size as #keys, where
 *         mask[i] = true if keys[i] is in vals;
 *                   false otherwise.
 *  */
template <typename T>
std::vector<bool> whereKeysInVals(const std::vector<T> &keys,
                                  std::vector<T> vals) {

  // to populate
  std::vector<bool> mask(keys.size(), false);

  // keys_[i] is initialized as (key[i], i), then keys_ is sorted.
  std::vector<std::tuple<T, uint64_t>> keys_(keys.size());
  for (uint64_t i = 0; i < keys.size(); ++i) {
    keys_[i] = {keys[i], i};
  }
  std::sort(keys_.begin(), keys_.end());

  // vals is sorted. We can now iterate through keys_ and vals, left to right
  std::sort(vals.begin(), vals.end());

  uint64_t keyIter{0};
  uint64_t valIter{0};

  while (keyIter < keys.size() && valIter < vals.size()) {
    while (std::get<0>(keys_[keyIter]) > vals[valIter] &&
           valIter < vals.size()) {
      ++valIter;
    }
    if (std::get<0>(keys_[keyIter]) == vals[valIter]) {
      mask[std::get<1>(keys_[keyIter])] = true;
    }
    ++keyIter;
  }

  return mask;
}

/**
 * \param ids. A vector of elements of type In.
 *
 * \param m. A Map, whose keys are of type In.
 *
 * \return A vector of the same length as #ids. The i'th element is either:
 *           if ids[i] is in m  : ids[i], implicitly cast to type Out.
 *           if ids[i] not in m : the default constructed Out.
 *
 * */
template <typename Out, typename Map, typename In>
std::vector<Out> whereIdsInMap(Map &&m, const std::vector<In> &ids) {
  std::vector<Out> outs(ids.size());
  for (uint64_t i = 0; i < ids.size(); ++i) {
    const auto found = m.find(ids[i]);
    if (found != m.cend()) {
      outs[i] = found->second;
    }
  }
  return outs;
}

/**
 * \return all of the values in #os, omiting all 'none's.
 * */
template <typename NonOptional, typename Optionals>
std::vector<NonOptional> nonOptionals(Optionals &&os) {
  std::vector<NonOptional> nonOptionals;
  for (auto &&o : os) {
    if (o.has_value()) {
      nonOptionals.push_back(o.value());
    }
  }
  return nonOptionals;
}

} // namespace util
} // namespace poprithms

#endif
