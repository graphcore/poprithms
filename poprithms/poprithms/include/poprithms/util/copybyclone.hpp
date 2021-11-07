// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_COPYBYCLONE_HPP
#define POPRITHMS_MEMORY_COPYBYCLONE_HPP

#include <memory>
namespace poprithms {
namespace util {

/**
 * A class to make 'default's from the rule-of-5 work for some classes
 * containing cloneable but uncopyable member variables.
 * https://en.cppreference.com/w/cpp/language/rule_of_three
 *
 * This class wraps a class T, where T has a 'clone' method which returns a
 * unique_ptr<T> or a unique_ptr<DerivedFromT>. Consider this motivating
 * example:
 *
 * <code>
 * struct Foo{
 *   std::unique_ptr<Foo> clone() const;
 * };
 *
 * struct Container{
 *   int id;
 *   std::unique_ptr<Foo> foo;
 * };
 * </code>
 *
 * Container (above) does not have a default copy-constructor, because
 * std::unique_ptr<Foo> is not copyable. We could implement a
 * copy-constructor, as
 *
 * <code>
 * Container(const Container & rhs): id(rhs.id), foo(rhs.foo.clone()) {}
 * </code>
 *
 * but this custom-code approach gets challenging if Container has many more
 * member variables. The CopyByClone class makes it simpler. Redefining
 * Container as
 *
 * <code>
 * struct Container{
 *   int id;
 *   CopyByClone<Foo> foo;
 * };
 * </code>
 *
 * we obtain the 'default' copy constructor for free, as well as the default
 * assignment operators and move constructor.
 *
 * Question: Why not store the #foo member by value?
 * Answer:   The use of unique_ptr might be required for polymorphism, or if
 *           the PIMPL design is being used.
 *
 * Question: Why not store the #foo member as a shared_ptr?
 * Answer:   A shared_ptr is the perfect solution if you don't want to clone
 *           the value #foo, only the pointer to the value. This is not always
 *           the desired behaviour though.
 * */

template <class T> class CopyByClone {
public:
  /** The pointer to the cloneable object which is being contained  */
  std::unique_ptr<T> uptr;

  CopyByClone();
  ~CopyByClone();

  CopyByClone(std::unique_ptr<T> x) : uptr(std::move(x)) {}

  CopyByClone(const CopyByClone<T> &rhs);

  // The move constructor does not throw. This is because
  // std::unique_ptr's move constructor is non-throwing.
  CopyByClone(CopyByClone<T> &&rhs) noexcept;

  CopyByClone<T> &operator=(const CopyByClone<T> &);
  CopyByClone<T> &operator=(CopyByClone<T> &&) noexcept;

  bool operator==(const CopyByClone<T> &rhs) const {
    return (uptr == nullptr && rhs.uptr == nullptr) ||
           (uptr != nullptr && rhs.uptr != nullptr && *uptr == *rhs.uptr);
  }
  bool operator!=(const CopyByClone<T> &rhs) const {
    return !operator==(rhs);
  }
};

} // namespace util
} // namespace poprithms

#endif
