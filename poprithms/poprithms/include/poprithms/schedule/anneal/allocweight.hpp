#ifndef POPRITHMS_SCHEDULE_ANNEAL_ALLOCWEIGHT
#define POPRITHMS_SCHEDULE_ANNEAL_ALLOCWEIGHT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>

namespace poprithms {
namespace schedule {
namespace anneal {

constexpr int NAW = 7;

// A generalization of an Alloc's size, AllocWeight is vectorized to allow for
// lexicographic comparisons. There are NAW "characters" to compare
// AllocWeights along, the centre of these, at (NAW - 1)/2 is used by default.
//
// Design note: Initially, this project used "using AllocWeight = double". But
// for integration into the existing popart scheduler, a work-around for
// priorities (both above and below Tensor size) was required. As a first
// work-around attempt, sufficiently large (small) doubles were used for these
// priorities. But numerical issues were difficult to avoid when very large
// and small doubles were compared, and this approach was decided to be
// unsustainable.
//
// Using a vectorized Alloc size such as the class below avoids numerical
// issues. Unfortunately, it is 2x slower (for recompute example with 200 Ops)
// than using just doubles and so:
// (TODO(T14826)) templatize for different AllocWeight.
//

class AllocWeight {
public:
  // An AllocWeight which is non-zero at 1 index.
  // relativeLexico should be in the range [ - NAW / 2 , + NAW / 2 ]
  // large negative relativeLexico values have priority in AllocWeight
  // comparisons
  AllocWeight(double _v_, int relativeLexico) : v{0, 0, 0, 0, 0, 0, 0} {
    // -3  -2  -1  0  1  2  3
    assert(relativeLexico > -(NAW + 1) / 2);
    assert(relativeLexico < (NAW + 1) / 2);
    v[static_cast<uint64_t>((NAW - 1) / 2 + relativeLexico)] = _v_;
  }

  // by default, the centre position is used.
  explicit AllocWeight(double _v_) : AllocWeight(_v_, 0) {}

  static AllocWeight zero() { return AllocWeight(0.0, 0); }
  static AllocWeight negativeOne() { return AllocWeight(-1.0, 0); }

  bool operator==(const AllocWeight &rhs) const { return v == rhs.v; }
  bool operator!=(const AllocWeight &rhs) const { return v != rhs.v; }
  bool operator<(const AllocWeight &rhs) const { return v < rhs.v; }
  bool operator>(const AllocWeight &rhs) const { return v > rhs.v; }

  double get(uint64_t i) { return v[i]; }

  AllocWeight getAbsolute() const {
    AllocWeight x = *this;
    for (uint64_t i = 0; i < NAW; ++i) {
      x.v[i] = std::abs(x.v[i]);
    }
    return x;
  }

  double getL1() const {
    double absSum{0.0};
    for (const auto &x : v) {
      absSum += std::abs(x);
    }
    return absSum;
  }

  AllocWeight &operator-=(const AllocWeight &rhs) {
    for (uint64_t i = 0; i < NAW; ++i) {
      v[i] -= rhs.v[i];
    }
    return *this;
  }

  AllocWeight &operator+=(const AllocWeight &rhs) {
    for (uint64_t i = 0; i < NAW; ++i) {
      v[i] += rhs.v[i];
    }
    return *this;
  }

  AllocWeight &operator+=(double b) {
    for (auto &x : v) {
      x += b;
    }
    return *this;
  }

  AllocWeight &operator/=(double d) {
    for (uint64_t i = 0; i < NAW; ++i) {
      v[i] /= d;
    }
    return *this;
  }

  AllocWeight &operator/=(AllocWeight d) {
    for (uint64_t i = 0; i < NAW; ++i) {
      if (d.get(i) != 0) {
        v[i] /= d.get(i);
      }
    }
    return *this;
  }

  AllocWeight &operator*=(double d) {
    for (uint64_t i = 0; i < NAW; ++i) {
      v[i] *= d;
    }
    return *this;
  }

  void append(std::ostream &ost) const {
    ost << '(' << v[0];
    for (auto i = 1; i < NAW; ++i) {
      ost << ',' << ' ' << v[i];
    }
    ost << ')';
  }

  std::string str() const {
    std::ostringstream oss;
    append(oss);
    return oss.str();
  }

  std::array<double, NAW> get() const { return v; }

private:
  std::array<double, NAW> v;
};

static AllocWeight operator*(int a, AllocWeight w) {
  AllocWeight b(w);
  b *= a;
  return b;
}

static AllocWeight operator*(AllocWeight w, int a) { return a * w; }

static AllocWeight operator+(const AllocWeight &a, const AllocWeight &b) {
  AllocWeight c(a);
  c += b;
  return c;
}

static AllocWeight operator+(double a, const AllocWeight &b) {
  AllocWeight c(b);
  c += a;
  return c;
}

static AllocWeight operator/(const AllocWeight &w, double d) {
  AllocWeight x(w);
  x /= d;
  return x;
}

static AllocWeight operator/(const AllocWeight &n, const AllocWeight &d) {
  AllocWeight x(n);
  x /= d;
  return x;
}

static AllocWeight operator-(const AllocWeight &a, const AllocWeight &b) {
  AllocWeight c(a);
  c -= b;
  return c;
}

static std::ostream &operator<<(std::ostream &ost, const AllocWeight &x) {
  x.append(ost);
  return ost;
}

static AllocWeight absolute(const AllocWeight &w) { return w.getAbsolute(); }

static double getL1(const AllocWeight &w) { return w.getL1(); }

static std::string toString(AllocWeight w) { return w.str(); }

using FallRate = AllocWeight;

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
