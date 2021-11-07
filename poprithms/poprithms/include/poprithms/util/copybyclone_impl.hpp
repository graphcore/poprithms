// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
  if (rhs.uptr) {
    this->uptr = rhs.uptr->clone();
  } else {
    this->uptr = nullptr;
  }
  return *this;
}

template <typename T>
CopyByClone<T> &CopyByClone<T>::operator=(CopyByClone<T> &&rhs) noexcept {
  if (this == &rhs) {
    return *this;
  }
  // if rhs.uptr == nullptr, this is still valid code:
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
    : uptr(rhs.uptr ? rhs.uptr->clone() : nullptr) {}

} // namespace util
} // namespace poprithms

#endif
