// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_COPYBYCLONE_IMPL_HPP
#define POPRITHMS_MEMORY_ALIAS_COPYBYCLONE_IMPL_HPP

#include <poprithms/util/copybyclone.hpp>
namespace poprithms {
namespace util {

template <typename T> CopyByClone<T>::CopyByClone()  = default;
template <typename T> CopyByClone<T>::~CopyByClone() = default;

// Assignment operators
//
template <typename T>
CopyByClone<T> &CopyByClone<T>::operator=(const CopyByClone<T> &rhs) {
  if (this == &rhs) {
    return *this;
  }
  this->uptr = rhs.uptr->clone();
  return *this;
}

template <typename T>
CopyByClone<T> &CopyByClone<T>::operator=(CopyByClone<T> &&rhs) noexcept {
  if (this == &rhs) {
    return *this;
  }
  this->uptr = std::move(rhs.uptr);
  return *this;
}

// Constructors
//
template <typename T>
CopyByClone<T>::CopyByClone(CopyByClone<T> &&rhs) noexcept
    : uptr(std::move(rhs.uptr)) {}

template <typename T>
CopyByClone<T>::CopyByClone(const CopyByClone<T> &rhs)
    : uptr(rhs.uptr->clone()) {}

} // namespace util
} // namespace poprithms

#endif
