// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/logging.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace poprithms {
namespace memory {
namespace nest {

namespace {

std::array<const Stripe *, 2> getPhaseOrdered(const Stripe &a,
                                              const Stripe &b) {
  if (a.phase() < b.phase()) {
    return {&a, &b};
  }
  return {&b, &a};
}

std::array<const Sett *, 2> getPeriodOrdered(const Sett &a, const Sett &b) {

  if (a.period() < b.period()) {
    return {&a, &b};
  }
  return {&b, &a};
}

std::array<const Sett *, 2> getDepthOrdered(const Sett &a, const Sett &b) {
  if (a.recursiveDepth_u64() < b.recursiveDepth_u64()) {
    return {&a, &b};
  }
  return {&b, &a};
}

// (longest on at depth 0, shortest on at depth 0)
std::array<const Sett *, 2> getOnOrdered(const Sett &a, const Sett &b) {
  if (a.atDepth(0).on() < b.atDepth(0).on()) {
    return {&a, &b};
  }
  return {&b, &a};
}

bool sameInDepthRange(const Sett &lhs,
                      const Sett &rhs,
                      uint64_t d0,
                      uint64_t d1) {
  for (uint64_t i = d0; i < d1; ++i) {
    if (lhs.atDepth(i) != rhs.atDepth(i)) {
      return false;
    }
  }
  return true;
}

int64_t greatestCommonDenominator(int64_t a, int64_t b) {
  if (a < 1 || b < 1) {
    throw error("gcd only computable for a,b > 0");
  }
  if (a < b) {
    std::swap(a, b);
  }
  // Euclid's algorithm. Uses that gcd(a, b) = gcd(a%b, b)
  int64_t t;
  while (b != 0ll) {
    t = a % b;
    a = b;
    b = t;
  }
  return a;
}

std::vector<Sett> insertPrefix(const Stripe &s, const std::vector<Sett> &ps) {
  std::vector<Sett> prefixed;
  prefixed.reserve(ps.size());
  for (const auto &p : ps) {
    std::vector<Stripe> stripes{s};
    stripes.insert(
        stripes.end(), p.getStripes().cbegin(), p.getStripes().cend());

    Sett nxt{stripes};
    if (!nxt.alwaysOff()) {
      prefixed.push_back(nxt);
    }
  }
  return prefixed;
}

} // namespace

Sett::Sett(const std::vector<Stripe> &s, bool canon) : stripes(s) {
  // repeatedly run the canonicalization pass until there is no change in the
  // depth of this Sett.
  if (canon) {
    auto l0 = recursiveDepth_u64();
    auto l1 = l0 + 1;
    while (l0 != 0 && l0 < l1) {
      canonicalize();
      l1 = l0;
      l0 = recursiveDepth_u64();
    }
  }
}

int64_t smallestCommonMultiple_i64(int64_t a, int64_t b) {
  auto gcdenom = greatestCommonDenominator(a, b);
  return a * b / gcdenom;
}

int64_t Sett::smallestCommonMultiple(const Sett &rhs) const {
  const int64_t period0 = period();
  const int64_t period1 = rhs.period();
  return smallestCommonMultiple_i64(period0, period1);
}

int64_t Sett::smallestCommonMultiple_v(const std::vector<Sett> &setts) {
  // smallest common multiple off all first Stripes in setts, or 1 if there
  // are no Stripes in Sett. Example
  // 11....11....11....11....11....11....11.... (repeats every 6)
  // .11.1..11.1..11.1..11.1..11.1..11.1..11.1. (repeats every 6)
  // 111111111111111111111111111111111111111111 (always on)
  // 1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1. (repeats every 2)
  // ..1...1...1...1...1...1...1...1...1...1... (repeats every 4)
  //
  // => returns 12.

  return std::accumulate(setts.cbegin(),
                         setts.cend(),
                         int64_t(1),
                         [](int64_t x, const Sett &pd1) {
                           return smallestCommonMultiple_i64(x, pd1.period());
                         });
}

bool Sett::equivalent(const Sett &rhs) const {

  if (getStripes() == rhs.getStripes()) {
    return true;
  }

  // compute the smallest period over which a comparison is required to ensure
  // equivalence:
  const auto end = smallestCommonMultiple(rhs);

  // Quick attempt at asserting non-equivalence: if different counts of '1's
  // over the range, then they must be different.
  const auto nToEnd = n(0, end);
  if (nToEnd != rhs.n(0, end)) {
    return false;
  }

  // At this point we know that they have the same number of '1's in the range
  // of interest. So we now just check that the intersection has the same
  // number of elements as rhs.
  const auto intersection = intersect(rhs);
  const auto nInter       = std::accumulate(
      intersection.cbegin(),
      intersection.cend(),
      0LL,
      [end](int64_t v, const Sett &x) { return v + x.n(0, end); });
  return nInter == nToEnd;
}

bool Sett::equivalent(const DisjointSetts &rhs) const {

  auto rhsScm = smallestCommonMultiple_v(rhs.get());
  int64_t scm = smallestCommonMultiple_i64(period(), rhsScm);

  // The number that are '1' (on) for this Sett
  const auto n0 = n(0, scm);

  // The number that on are '1' (on) for the Setts in rhs
  std::vector<int64_t> rhsNs;
  rhsNs.reserve(rhs.size());
  for (const auto &pRhs : rhs.get()) {
    rhsNs.push_back(pRhs.n(0, scm));
  }

  if (std::accumulate(rhsNs.cbegin(), rhsNs.cend(), 0) != n0) {
    return false;
  }

  // We know that rhs are disjoint (by assumption of function) and
  // have enough '1', we now check that the '1's are only where this has '1'.
  return std::all_of(rhs.cbegin(), rhs.cend(), [this](const Sett &x) {
    return contains(x);
  });
}

bool DisjointSetts::equivalent(const DisjointSetts &rhs) const {

  // step 1 remove identical Setts in this and rhs, the comparison can be done
  // without them.
  auto a         = get();
  auto b         = rhs.get();
  const auto cmp = [](const Sett &lhs_, const Sett &rhs_) {
    return lhs_.getStripes() < rhs_.getStripes();
  };
  std::sort(a.begin(), a.end(), cmp);
  std::sort(b.begin(), b.end(), cmp);
  std::vector<Sett> aReduced;
  std::vector<Sett> bReduced;
  auto aIter = a.cbegin();
  auto bIter = b.cbegin();
  while (aIter != a.end() && bIter != b.end()) {
    if (cmp(*aIter, *bIter)) {
      aReduced.push_back(*aIter);
      ++aIter;
    } else if (cmp(*bIter, *aIter)) {
      bReduced.push_back(*bIter);
      ++bIter;
    }
    // Identical Stripes in the 2 Setts:
    else {
      ++aIter;
      ++bIter;
    }
  }
  aReduced.insert(aReduced.end(), aIter, a.cend());
  bReduced.insert(bReduced.end(), bIter, b.cend());

  // Step 2: try to deduce non-equivalence by considering the total number of
  // ons
  const auto scm =
      smallestCommonMultiple_i64(Sett::smallestCommonMultiple_v(aReduced),
                                 Sett::smallestCommonMultiple_v(bReduced));
  DisjointSetts lhsReduced(aReduced);
  DisjointSetts rhsReduced(bReduced);
  const auto lhsCounts = lhsReduced.totalOns(scm);
  const auto rhsCounts = rhsReduced.totalOns(scm);
  if (lhsCounts != rhsCounts) {
    return false;
  }

  // Step 3: we have deduced that the that total counts are the same, now just
  // need to confirm that one is contained in the other:
  for (const auto &lSett : lhsReduced) {
    if (!lSett.containedIn(rhsReduced)) {
      return false;
    }
  }

  return true;
}

int64_t DisjointSetts::totalOns(int64_t end) const {
  return std::accumulate(
      setts_.cbegin(),
      setts_.cend(),
      int64_t(0),
      [end](int64_t c, const auto &sett) { return c + sett.n(0, end); });
}

void Sett::confirmEquivalent(const Sett &rhs) const {
  confirmEquivalent(DisjointSetts(rhs));
}

void Sett::confirmEquivalent(const DisjointSetts &rhs) const {

  confirmDisjoint(rhs.get());
  auto equiv = equivalent(rhs);

  if (equiv && !log().shouldLogTrace()) {
    return;
  }

  std::ostringstream oss;
  oss << "In Sett::confirmEquivalent(.)\n";
  oss << *this << ".equivalent(";
  for (const auto &x : rhs.get()) {
    oss << "\n            ";
    x.append(oss);
  }
  oss << ") is " << equiv << ".";
  auto debugString = oss.str();

  log().trace(debugString);

  if (equiv) {
    return;
  }

  throw error(debugString);
}

std::array<Sett, 2> Sett::getPeriodSplit() const {
  if (!hasStripes()) {
    throw error(
        "Internal Logic Error: unexpected case A in Sett::getPeriodSplit");
  }

  const auto &s0 = atDepth(0);

  if (s0.phase() + s0.on() <= s0.period()) {
    throw error(
        "Internal Logic Error: unexpected case B in Sett::getPeriodSplit");
  }

  // The non-trivial case, where phase is large, and "on" crosses period
  // bound
  //                 period
  //                   |
  //                   |
  //  on  off     on
  // 1111.....1111111111
  //
  // .........1111111111   rhs
  // =========             rhs off   : phase
  // =========             rhs phase : phase
  //          ==========   rhs on    : period - phase
  //
  // 1111...............   lhs
  // ====                  lhs on    : phase - off
  //     ===============   lhs off   : period + off - phase
  //

  const auto rhsOn    = s0.period() - s0.phase();
  const auto rhsPhase = s0.phase();
  const auto rhsOff   = s0.phase();
  std::vector<Stripe> rhs{{rhsOn, rhsOff, rhsPhase}};

  const auto lhsOn    = s0.phase() - s0.off();
  const auto lhsOff   = s0.period() + s0.off() - s0.phase();
  const auto lhsPhase = 0;
  std::vector<Stripe> lhs{{lhsOn, lhsOff, lhsPhase}};

  if (recursiveDepth() == 1) {
    return {Sett{lhs}, Sett{rhs}};
  }

  rhs.insert(rhs.end(), stripes.cbegin() + 1, stripes.cend());

  const auto &s1 = atDepth(1);
  lhs.push_back({s1.on(), s1.off(), s1.phase() - rhsOn});
  lhs.insert(lhs.end(), stripes.cbegin() + 2, stripes.cend());

  return {Sett{lhs}, Sett{rhs}};
}

int64_t Sett::getOn(int64_t nth) const {

  if (alwaysOff()) {
    throw error("Call to getOn for empty Sett");
  }

  if (nth < 0) {

    // xxx....xxx....xxx....xxx
    //             432101234
    //             ---------
    const auto nOnInPeriod = n(0, period());

    // nth + phi*nOnInPeriod > 0
    // phi > -nth / nOnInPeriod
    // phi = -nth / nOnInPeriod + 1
    const auto phi = -nth / nOnInPeriod + 1;
    return getOnRecurse(nth + phi * nOnInPeriod) - phi * period();
  }

  auto onth = getOnRecurse(nth);
  return onth;
}

int64_t Sett::getOnRecurse(int64_t nOn) const {

  if (!hasStripes()) {
    return nOn;
  }
  const Stripe &stripe0 = atDepth(0ul);
  int64_t nInPrefix0    = n(0, stripe0.phase());

  // ends before first start of stripe0
  if (nOn < nInPrefix0) {
    if (recursiveDepth_u64() == 1) {
      return nOn;
    }
    std::vector<Stripe> subStripes;
    subStripes.reserve(recursiveDepth_u64() - 1);
    subStripes.push_back(
        {atDepth(1ul).on(),
         atDepth(1ul).off(),
         atDepth(1ul).phase() - (stripe0.period() - stripe0.phase())});
    for (uint64_t d = 2; d < recursiveDepth_u64(); ++d) {
      subStripes.push_back(atDepth(d));
    }
    return Sett(subStripes).getOnRecurse(nOn);
  }

  // nOn >= nInPrefix0
  int64_t nInFull0 = n(stripe0.phase(), stripe0.phase() + stripe0.period());
  auto nComplete0  = (nOn - nInPrefix0) / nInFull0;
  auto nRemaining  = nOn - nInPrefix0 - nComplete0 * nInFull0;
  return stripe0.phase() + nComplete0 * stripe0.period() +
         fromDepth(1).getOnRecurse(nRemaining);
}

Sett Sett::getReverse(int64_t x) const {
  if (!hasStripes()) {
    return createAlwaysOn();
  }
  std::vector<Stripe> revStripes;
  revStripes.reserve(recursiveDepth_u64());
  for (uint64_t depth = 0UL; depth < recursiveDepth_u64(); ++depth) {
    const Stripe &stripe = atDepth(depth);
    int64_t newPhase     = x - stripe.lastStartNotAfter(x) + stripe.off();
    revStripes.push_back({stripe.on(), stripe.off(), newPhase});
    x = atDepth(depth).on();
  }

  return revStripes;
}

std::vector<int> Sett::getRepeatingOnCount(const std::vector<Sett> &setts) {
  const auto end = smallestCommonMultiple_v(setts);
  std::vector<int> counts(static_cast<uint64_t>(end), 0);
  for (const auto &x : setts) {
    auto ons = x.getOns(0, end);
    for (auto v : ons) {
      counts[static_cast<uint64_t>(v)] += 1;
    }
  }
  return counts;
}

bool Sett::disjoint(const std::vector<Sett> &setts) {
  // There is probably some non-trivial data-structure to store Setts in which
  // permits an o(N^2) algorithm for this. To be investigated if this becomes
  // a bottleneck.
  for (auto iter0 = setts.cbegin(); iter0 != setts.cend(); ++iter0) {
    for (auto iter1 = std::next(iter0); iter1 != setts.cend(); ++iter1) {
      if (!iter0->disjoint(*iter1)) {
        return false;
      }
    }
  }
  return true;
}

bool Sett::alwaysOff() const {
  if (!hasStripes()) {
    return false;
  }
  // If this Sett was created with canonicalization on, we can just check the
  // first Stripe. But if it is not, we need to check recursively for
  // emptiness:
  return n(0, atDepth(0).period()) == 0;
}

bool Sett::disjoint(const Sett &rhs) const {

  if (alwaysOff() || rhs.alwaysOff()) {
    return true;
  }

  if (!rhs.hasStripes() || !hasStripes()) {
    return false;
  }

  auto scm                = smallestCommonMultiple(rhs);
  auto nIntersectApproach = scm / period();
  auto nFillApproach      = rhs.period() * period() / scm;

  // compute intersection, and return true if empty.
  if (nIntersectApproach < nFillApproach) {
    const auto inter = intersect(rhs);
    return std::all_of(inter.cbegin(), inter.cend(), [](const Sett &x) {
      return x.alwaysOff();
    });
  }

  // It is easy to see that non-empty Setts of co-prime periods are not
  // disjoint, this approach is a generalization of that approach to
  // almost-co-prime Setts. It is essentially using the Chinese Remainder
  // Theorem: https://en.wikipedia.org/wiki/Chinese_remainder_theorem
  //
  // Example:
  //
  // lhs :  .1...1.1...1    (period = 12, only on at mod(2) = 1)
  // rhs :  1.1...1...1.1.  (period = 14, only on at mod(2) = 0)
  //
  // The above are disjoint, as can be seen by partitioning by
  // mod(2).
  else {
    for (int64_t i = 0; i < nFillApproach; ++i) {
      auto lhsFilled = sampleAt({{{1, nFillApproach - 1, i}}});
      auto rhsFilled = rhs.sampleAt({{{1, nFillApproach - 1, +i}}});
      auto lhsFilledNonEmpty =
          std::any_of(lhsFilled.cbegin(),
                      lhsFilled.cend(),
                      [](const Sett &sett) { return !sett.alwaysOff(); });

      auto rhsFilledNonEmpty =
          std::any_of(rhsFilled.cbegin(),
                      rhsFilled.cend(),
                      [](const Sett &sett) { return !sett.alwaysOff(); });

      if (lhsFilledNonEmpty && rhsFilledNonEmpty) {
        return false;
      }
    }
    return true;
  }
}

bool Sett::contains(const Sett &rhs) const {

  const auto n0      = smallestCommonMultiple(rhs);
  const auto nOnRhs  = rhs.n(0, n0);
  const auto nOnThis = n(0, n0);

  // If this is always on, it contains any Sett
  if (nOnThis == n0) {
    return true;
  }

  // At this point this has been confirmed to not always be on. If rhs is
  // always on, then this does not contain it.
  if (nOnRhs == n0) {
    return false;
  }

  // every set contains the empty set
  if (nOnRhs == 0) {
    return true;
  }

  // At this point, we know that rhs is on at least at 1 index. So if this is
  // always off, it cannot contain rhs.
  if (nOnThis == 0) {
    return false;
  }

  // At this point, we know that both Setts are sometimes on, and sometimes
  // off, and so have at least 1 Stripe each. If they have co-prime periods,
  // then each on in rhs will align at some point with an off in this, so
  // cannot be contained.
  if (n0 == atDepth(0).period() * rhs.atDepth(0).period()) {
    return false;
  }

  // No more "tricks" to conclude with, so will check that the number of ons
  // in the intersection with rhs is the same as the number of ons in rhs.
  const auto inter  = intersect(rhs);
  auto nOnIntersect = std::accumulate(
      inter.cbegin(), inter.cend(), 0LL, [n0](int64_t cnt, const Sett &x) {
        return x.n(0, n0) + cnt;
      });
  return nOnIntersect == nOnRhs;
}

bool Sett::containedIn(const DisjointSetts &rhs) const {

  auto scm = smallestCommonMultiple_v(rhs.get());
  scm      = smallestCommonMultiple_i64(scm, period());

  int64_t intersize{0};
  for (const auto &r : rhs.get()) {
    const auto inter = intersect(r);
    for (const auto &z : inter.get()) {
      intersize += z.n(0LL, scm);
    }
  }

  if (intersize != n(0L, scm)) {
    return false;
  }

  return true;
}

void Sett::canonicalize() {

  // TODO(T23328) more cases to hit.

  std::vector<Stripe> reverseCanonical;
  for (uint64_t d = recursiveDepth_u64(); d != 0; --d) {
    const Stripe &child = stripes[d - 1];

    // child is always off : Sett is always off.
    if (child.off() == child.period()) {

      stripes = {Stripe(0, // on
                        1, // off
                        0  // phase
                        )};
      return;
    }

    // child aways on, can't do anything simple (unless check periods)
    if (child.alwaysOn() && reverseCanonical.empty()) {
      // child is quietly removed
      continue;
    }

    // Case to consider: periods must align.
    // 111111111111111111111 (5,0,2)
    //   1  1 1  1 1  1 1  1  (1,2,0)
    //      !=
    //   1  1  1  1  1  1  1  (1,2,2)
    if (!reverseCanonical.empty()) {

      // .1111111...1111111...1111111...
      // ..1....1....1....1....1....1...
      // In this example, the Sett is ((7,3,1),(1,4,1), and
      // it can be replace with (1,4,2).
      auto b = reverseCanonical.back();
      if (child.period() % b.period() == 0 // periods align
          && b.nOn(0, child.period()) ==
                 b.nOn(0, child.on()) // no ons inserted by removing child
      ) {
        reverseCanonical.back() = {
            b.on(), b.off(), b.phase() + child.phase()};
        continue;
      }
    }

    // child has a parent, check for merging of child and parent
    if (d != 1) {
      Stripe &parent = stripes[d - 2];
      // the child is never on while the parent is on: stripeless.
      if (child.phase() >= parent.on() && child.phase() <= child.off()) {
        stripes = {Stripe(0, 1, 0)};
        return;
      }

      if (child.phase() == 0 && child.on() >= parent.on()) {
        // 1111    parent
        // 111111  child
        // child is quietly removed
        continue;
      }

      // .....11.....             parent
      //...111111.....111111....  child
      if (child.phase() - child.off() >= parent.on()) {
        if (!reverseCanonical.empty()) {
          const auto b            = reverseCanonical.back();
          reverseCanonical.back() = {
              b.on(), b.off(), b.phase() - (child.period() - child.phase())};
        }
        continue;
      }

      //
      // 11111111111111.......    parent
      // .....11111......11111   child
      //
      if (child.phase() < parent.on() && //
          child.phase() <= child.off() &&
          child.phase() + child.period() >= parent.on()) {

        auto newParentOn = std::min(parent.on() - child.phase(), child.on());
        auto newParentPhase = parent.phase() + child.phase();
        auto newParentOff   = parent.period() - newParentOn;
        parent = Stripe(newParentOn, newParentOff, newParentPhase);
        continue;
      }

      // Note that this optimization only works when child is the base stripe,
      // (1,0,0)(3,0,2)(1,1,0) -> (1,0,0)(1,1,0) caught me out. For the case
      // where child is not the base stripe, we need to adjust the
      // grand-childs phase. An outstanding task.
      //
      // ....111111111........  parent
      // ..1111...........1111  child

      if (child.phase() >= parent.on() && reverseCanonical.empty()) {
        auto newParentOn = std::max<int64_t>(
            0ll, std::min(parent.on(), child.phase() - child.off()));
        auto newParentOff   = parent.period() - newParentOn;
        auto newParentPhase = parent.phase();
        parent = Stripe(newParentOn, newParentOff, newParentPhase);
        continue;
      }

      // Example case: (11,1,0)(3,1,0)
      // 11111111111. parent
      // 111.111.111. child
      if (parent.period() % child.period() == 0 &&
          child.phase() - child.off() <= 0 &&
          child.firstStartNotBefore(parent.period()) - child.off() <=
              parent.on()) {
        // parent silently replaced by (phase adjusted) child
        parent = {child.on(), child.off(), child.phase() + parent.phase()};
        continue;
      }
    }

    reverseCanonical.push_back(child);
  }

  stripes = {};
  stripes.reserve(reverseCanonical.size());
  for (auto iter = reverseCanonical.crbegin();
       iter != reverseCanonical.crend();
       ++iter) {
    stripes.push_back(*iter);
  }
}

void Sett::append(std::ostream &ost) const {
  ost << '(';
  for (const auto &stripe : stripes) {
    stripe.append(ost);
  }
  ost << ')';
}

Sett Sett::fromDepth(int d) const {
  if (d > recursiveDepth()) {
    std::ostringstream oss;
    oss << "Call to Sett::fromDepth(" << d << ") is not valid for "
        << "Sett of recursive depth " << recursiveDepth();
    throw error(oss.str());
  }
  return Sett({stripes.cbegin() + d, stripes.cend()});
}

DisjointSetts Sett::intersect(const Sett &lhs, const Sett &rhs) {
  auto scm          = lhs.smallestCommonMultiple(rhs);
  auto intersection = intersectRecurse(lhs,
                                       rhs,
                                       0,  // depth
                                       0,  // lowCorrect
                                       scm // uppCorrect
  );
  std::vector<Sett> polishedAndPrefixed;
  for (auto &x : intersection.get()) {
    x = x.adjustedPrepend({scm, 0, 0});
    if (!x.alwaysOff()) {
      polishedAndPrefixed.push_back(x);
    }
  }
  return DisjointSetts(polishedAndPrefixed);
}

DisjointSetts Sett::intersectRecurse(const Sett &lhsIn,
                                     const Sett &rhsIn,
                                     uint64_t depth,
                                     int64_t low,
                                     int64_t upp) {

  if (lhsIn.n(low, upp) == 0 || rhsIn.n(low, upp) == 0) {
    return {};
  }

  // intersection with full set is other set
  if (!rhsIn.hasStripes()) {
    return {lhsIn};
  }

  // intersection with full set is other set (in other direction)
  if (!lhsIn.hasStripes()) {
    return {rhsIn};
  }

  // lhs has smaller period (or equal) than rhs from here on in:
  auto periodSorted = getPeriodOrdered(lhsIn, rhsIn);
  const auto &lhs   = *std::get<0>(periodSorted);
  const auto &rhs   = *std::get<1>(periodSorted);

  const auto &l0 = lhs.atDepth(0);
  const auto &r0 = rhs.atDepth(0);

  if (l0.period() == r0.period()) {
    const auto period0 = l0.period();

    if (l0.phase() == r0.phase()) {
      const auto phase0 = l0.phase();

      //
      // intersect a sparse version of shorter with longer from depth 1, and
      // insert prefix as a final step.
      //
      // longer   ..1111111..1111111..
      // shorter  ..111......111......
      //
      auto onOrdered        = getOnOrdered(lhs, rhs);
      const auto &shorter   = *std::get<0>(onOrdered);
      const auto &longer    = *std::get<1>(onOrdered);
      const auto subLonger  = longer.fromDepth(1);
      const auto subShorter = shorter.fromDepth(1);

      const auto shortOn = shorter.atDepth(0).on();
      Stripe stripe0{shortOn, period0 - shortOn, phase0};
      auto subInter = intersectRecurse(subShorter,
                                       subLonger,
                                       depth + 1,
                                       0,      // lowCorrect
                                       shortOn // uppCorrect
      );
      for (auto &x : subInter) {
        x.prependStripes({stripe0});
      }
      return subInter;
    } else {

      // Consider two Setts of equal period, imagine the period is 24 hours.
      // Suppose one Sett is the "day worker" the other the "night shifter".
      // When the day worker arrives, the night worker might still be there.
      // Similarly, when the day worker arrives, the night worker might still
      // be there. We denote these durations as t(day,night) and t(night,day).
      //
      // At this point in the code the 2 Setts do not have the same phase (do
      // not start shifts at the same time) but do have the same period. So,
      //
      // union(t(day,night), t(night,day)) = intersection(day,night).
      //
      // We use this decomposition to reduce to the previous case of equal
      // phases.
      //
      // Example 1:
      // 11111.....11111.....11111.....11111.... day
      // ...11111111..11111111..11111111..11111  night
      //    11        11        11        11     day->night
      //           1         1         1         night->day.
      //
      // Example 2 (the day-night analogy is not perfect!)
      // 111..1111111111..1111111111..1111111111... day
      // .......111.........111.........111........ night
      //        111         111         111         day->night
      //                                            night->day = empty

      std::vector<std::array<const Sett *, 2>> transitions{{&lhs, &rhs},
                                                           {&rhs, &lhs}};
      std::vector<Sett> extended;
      for (const auto &[pday, pnight] : transitions) {
        const auto &d0 = pday->atDepth(0);
        const auto &n0 = pnight->atDepth(0);

        // duration from night shift arriving to day shift ending
        auto eodOn =
            d0.phase() + d0.on() - n0.firstStartNotBefore(d0.phase());
        if (eodOn > 0 && eodOn < d0.on()) {
          auto endOfDay =
              pday->adjustedPrepend({eodOn, d0.period() - eodOn, n0.phase()});
          auto transitionIntersect =
              intersectRecurse(endOfDay,
                               *pnight,
                               depth + 1,
                               n0.phase(),        // lowCorrect
                               n0.phase() + eodOn // uppCorrect
              );

          extended.insert(extended.end(),
                          transitionIntersect.cbegin(),
                          transitionIntersect.cend());
        }
      }
      return DisjointSetts(std::move(extended));
    }
  }

  // different periods. Recall rhs has longer period, we sparsify it and
  // intersect the sparse version's sub-Stripes with lhs.
  else {
    auto scm           = smallestCommonMultiple_i64(l0.period(), r0.period());
    auto rhsReplFactor = scm / r0.period();
    auto rhsSub        = rhs.fromDepth(1);
    std::vector<Sett> intersection{};
    for (int64_t rhsIndex = 0; rhsIndex < rhsReplFactor; ++rhsIndex) {
      const auto superPhase = rhsIndex * r0.period() + r0.phase();
      const Stripe prefix{r0.on(), scm - r0.on(), superPhase};
      if (!prefix.allOff(low, upp)) {
        auto leftShifted     = lhs.phaseShifted(-superPhase);
        auto subIntersection = intersectRecurse(rhsSub,
                                                leftShifted,
                                                depth + 1,
                                                0,      // lowCorrect
                                                r0.on() // uppCorrect
        );
        intersection.reserve(subIntersection.size() + intersection.size());
        for (auto &subSett : subIntersection) {
          subSett.prependStripes({prefix});
        }
        intersection.insert(intersection.end(),
                            subIntersection.cbegin(),
                            subIntersection.cend());
      }
    }
    return DisjointSetts(std::move(intersection));
  }
}

int64_t Sett::find(int64_t begin) const {

  if (begin < 0) {
    // begin + phi*period() > 0
    // phi*period() > -begin
    // phi > -begin/period()
    // phi = -begin/period() + 1

    const auto phi          = -begin / period() + 1;
    const auto shiftedBegin = begin + phi * period();
    auto delta              = find(shiftedBegin) - shiftedBegin;
    return begin + delta;
  }
  // The number of ons in [0, begin):
  auto tilBegin = n(begin);
  return getOn(tilBegin);
}

// result must be correct in range [lowX, highX)
DisjointSetts Sett::sampleRecurse(const Sett &x,
                                  const Sett &indices,
                                  int depth,
                                  int64_t lowX,
                                  int64_t uppX) {

  // xxxx....x.xxx....xxxx....xxx.x
  // 1..1...111...11...11..
  // 0  1   234   56   78
  //    |          |
  // [lowX, uppX) = [1,6)   =>
  // [lowIndex, uppIndex) = [3,14).

  if (!indices.hasStripes()) {
    return {x};
  }

  const auto lowIndex = indices.getOn(lowX);
  const auto uppIndex = indices.getOn(uppX);

  const auto nOnInRequiredRange_x = x.n(lowIndex, uppIndex);

  if (nOnInRequiredRange_x == 0) {
    return {};
  }

  if (nOnInRequiredRange_x == uppIndex - lowIndex) {
    return {Sett::createAlwaysOn()};
  }

  const auto &x0 = x.atDepth(0);
  const auto &i0 = indices.atDepth(0);

  const auto nChosen0 = indices.n(0, i0.period());
  auto foo            = indices.getOns(0, i0.period());

  if (nChosen0 == 0) {
    std::ostringstream oss;
    oss << "Invalid indices in sample, indices (" << indices
        << ") is not permitted to be always off ("
        << "this is kind of equivalent to division by 0"
        << ").";
    throw error(oss.str());
  }

  if (i0.phase() != 0) {

    const auto delta     = indices.n(0, i0.phase());
    auto x_shifted       = x.phaseShifted(-i0.phase());
    auto indices_shifted = indices.phaseShifted(-i0.phase());

    // I do not think this is incorrect:
    auto sampled = sampleRecurse(
        x_shifted, indices_shifted, depth + 1, lowX - delta, uppX - delta);

    for (auto &s : sampled) {
      s.shiftPhase(delta);
    }

    return sampled;

  }

  // at this point, we have ensured i0.phase() == 0.
  //
  else if (x0.period() != i0.period()) {

    auto scm         = smallestCommonMultiple_i64(x0.period(), i0.period());
    auto iReplFactor = scm / i0.period();
    auto xReplFactor = scm / x0.period();

    std::vector<Sett> sampled{};

    int64_t iIndex = 0;
    while (iIndex < iReplFactor) {

      auto indicesPhase = iIndex * i0.period() + i0.phase();

      auto indicesSparse       = indices;
      indicesSparse.stripes[0] = {i0.on(),
                                  scm - i0.on(), // off
                                  indicesPhase};

      const Stripe pref{
          nChosen0, (iReplFactor - 1) * nChosen0, iIndex * nChosen0};

      if (pref.allOff(lowX, uppX)) {
        ++iIndex;
        continue;
      }

      // to save on the inner loop, using sparse sampling.
      const auto &iSparse    = indicesSparse.stripes[0];
      const auto xFirstStart = iSparse.phase() - x0.on();
      const auto xFirstIndex = (xFirstStart - x0.phase()) / x0.period();
      const auto xLastStart  = iSparse.phase() + iSparse.on();
      const auto xLastIndex  = xLastStart / x0.period() + 1;

      for (int64_t xIndex = xFirstIndex;
           xIndex < std::min(xLastIndex, xFirstIndex + xReplFactor);
           ++xIndex) {
        // the previous less efficient O(N^2) approach:
        // for (int64_t xIndex = 0; xIndex < xReplFactor; ++xIndex)
        auto xSparse       = x;
        xSparse.stripes[0] = {
            x0.on(),                          // on
            scm - x0.on(),                    // off
            xIndex * x0.period() + x0.phase() // phase
        };

        auto sparseSampled =
            sampleRecurse(xSparse, indicesSparse, depth + 1, 0, nChosen0);
        for (auto sett : insertPrefix(pref, sparseSampled.get())) {
          sampled.push_back(sett);
        }
      }

      // A special case optimization (can be removed) for super-incrementing
      // iIndex
      if (x0.period() % i0.period() == 0) {
        // what the next indices phase would be if we went through every phase
        auto nextIndicesPhase = indicesPhase + i0.period();

        // starting from the next phase, where is the next position where
        // indices actually might capture a non-0 x?
        // Recall, x.find(i) is the first index >= i where x is '1'.
        nextIndicesPhase   = x.find(nextIndicesPhase) - i0.period();
        auto proposedDelta = (nextIndicesPhase - indicesPhase) / i0.period();
        auto nextIndex     = iIndex + std::max<int64_t>(1LL, proposedDelta);
        iIndex             = nextIndex;
      } else {
        ++iIndex;
      }
    }
    return DisjointSetts(std::move(sampled));
  }

  // at this point, we have ensured that
  // 1) i0.phase() == 0, and
  // 2) x0.period() == i0.period()
  else {

    const auto period0 = x0.period();
    if (x0.phase() + x0.on() > period0) {
      auto x_split    = x.getPeriodSplit();
      const auto &x_l = std::get<0>(x_split);
      const auto &x_r = std::get<1>(x_split);

      auto sample_l = sampleRecurse(x_l, indices, depth + 1, lowX, uppX);
      auto sample_r = sampleRecurse(x_r, indices, depth + 1, lowX, uppX);

      sample_l.get().insert(
          sample_l.end(), sample_r.cbegin(), sample_r.cend());
      return sample_l;
    }

    // -> i0.phase() == 0
    // -> x0.period() == i0.period()
    // -> x0.phase() + x0.on() <= x0.period()
    else {
      Stripe x_r_stripe_0{i0.on(), 0, 0};
      std::vector<Stripe> x_r_stripes{x_r_stripe_0};
      x_r_stripes.reserve(x.recursiveDepth_u64() + 1);
      x_r_stripes.insert(
          x_r_stripes.end(), x.getStripes().cbegin(), x.getStripes().cend());

      std::vector<Stripe> indices_r_stripes(
          std::next(indices.getStripes().cbegin()),
          indices.getStripes().cend());

      Sett x_r{x_r_stripes};
      Sett indices_r{indices_r_stripes};
      auto sampled = sampleRecurse(x_r, indices_r, depth + 1, 0, nChosen0);
      for (auto &sampledPart : sampled) {
        sampledPart.prependStripes({{nChosen0, 0, 0}});
      }
      return sampled;
    }
  }
}

DisjointSetts Sett::intersect(const Sett &rhs) const {
  return intersect(*this, rhs);
}

DisjointSetts Sett::sampleAt(const Sett &indices) const {
  auto filtered = sample(*this, indices);
  return filtered;
}

int64_t Sett::nRecursive(uint64_t depth, int64_t begin, int64_t end) const {

  if (stripes.size() == depth) {
    return end - begin;
  }

  int64_t total = 0;

  const Stripe &current = stripes[depth];

  // begin -> a0 -> a1 -> end (a1 may be less than a0)
  auto a0 = current.firstStartNotBefore(begin);
  auto a1 = current.lastStartNotAfter(end);

  // begin -> a0
  if (a0 - begin > current.off()) {
    total +=
        nRecursive(depth + 1, current.period() - (a0 - begin), current.on());
  }

  // a0 -> a1
  total +=
      (a1 - a0) / current.period() * nRecursive(depth + 1, 0LL, current.on());

  // a1 -> end
  total += nRecursive(depth + 1, 0LL, std::min(current.on(), end - a1));
  return total;
}

int64_t Sett::n(int64_t start, int64_t end) const {

  if (start < 0) {
    // start + phi*period() > 0
    // phi > -start/period();
    // phi = -start/period() + 2
    const auto phi = -start / period() + 2;
    return n(start + phi * period(), end + phi * period());
  }
  int64_t pn = hasStripes() ? nRecursive(0, start, end) : end - start;
  return pn;
}

std::vector<int64_t> Sett::getOns(int64_t start, int64_t end) const {

  if (start > end) {
    std::ostringstream oss;
    oss << "In Sett::getOns(start=" << start << ", end=" << end
        << "). This is not allowed, "
        << "start cannot be greater than end in this function.";
    throw error(oss.str());
  }
  if (start < 0) {
    std::ostringstream oss;
    oss << "At this time, Sett::getOns(start, end) "
        << "requires start >= 0. "
        << " (this is easy to fix, see for example "
        << "Sett::find and Sett::getOn).";
    throw error(oss.str());
  }

  auto ons = getOnsRecurse(0UL, start, end);

  return ons;
}

Sett Sett::adjustedPrepend(const Stripe &prefix) const {
  std::vector<Stripe> prependedStripes{prefix};
  prependedStripes.reserve(recursiveDepth_u64() + 1);
  if (hasStripes()) {
    auto on    = atDepth(0).on();
    auto off   = atDepth(0).off();
    auto phase = atDepth(0).phase() - prefix.phase();
    prependedStripes.push_back({on, off, phase});
  }
  for (uint64_t si = 1; si < recursiveDepth_u64(); ++si) {
    prependedStripes.push_back(atDepth(si));
  }
  return Sett{prependedStripes};
}

std::vector<int64_t>
Sett::getOnsRecurse(uint64_t depth, int64_t start, int64_t end) const {

  if (start == end) {
    return {};
  }

  if (depth == recursiveDepth_u64()) {
    std::vector<int64_t> offsets(static_cast<uint64_t>(end - start));
    std::iota(offsets.begin(), offsets.end(), start);
    return offsets;
  }

  const auto &stripe0        = atDepth(depth);
  const auto nextStripeBegin = stripe0.firstStartNotBefore(start);

  if (stripe0.allOff(start, end)) {
    return {};
  }

  if (stripe0.allOn(start, end)) {
    const auto subStart = stripe0.lastStartNotAfter(start);
    auto ons = getOnsRecurse(depth + 1, start - subStart, end - subStart);
    for (auto &on : ons) {
      on += subStart;
    }
    return ons;
  }

  // prefix
  auto preStart  = start;
  auto preEnd    = nextStripeBegin;
  auto preLength = preEnd - preStart;
  std::vector<int64_t> preOffsets;

  // main
  auto mainStart    = preEnd;
  auto nMainStripes = (end - mainStart) / stripe0.period();
  auto mainLength   = stripe0.period() * nMainStripes;
  auto mainEnd      = mainStart + mainLength;
  std::vector<int64_t> mainOffsets;

  // suffix
  auto postStart  = mainEnd;
  auto postEnd    = std::min(postStart + stripe0.on(), end);
  auto postLength = postEnd - postStart;
  std::vector<int64_t> postOffsets;

  // populate prefix offsets
  if (preLength > stripe0.off()) {
    auto localStart = start - (nextStripeBegin - stripe0.period());
    auto localEnd   = localStart + (preLength - stripe0.off());
    preOffsets      = getOnsRecurse(depth + 1, localStart, localEnd);
    for (auto &x : preOffsets) {
      x += (start - localStart);
    }
  }

  // populate main offsets
  if (nMainStripes > 0) {
    auto localOffsets = getOnsRecurse(depth + 1, 0, stripe0.on());
    for (int64_t i = 0; i < nMainStripes; ++i) {
      int64_t globalOffset = mainStart + stripe0.period() * i;
      for (auto x : localOffsets) {
        mainOffsets.push_back(x + globalOffset);
      }
    }
  }

  // populate suffix offsets
  if (postLength > 0) {
    postOffsets = getOnsRecurse(depth + 1, 0, postLength);
    for (auto &x : postOffsets) {
      x += postStart;
    }
  }

  std::vector<int64_t> offsets;
  for (const std::vector<int64_t> &x :
       {preOffsets, mainOffsets, postOffsets}) {
    offsets.insert(offsets.end(), x.begin(), x.end());
  }
  return offsets;
}

std::ostream &operator<<(std::ostream &ost, const Sett &sett) {
  sett.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const OptionalSett1 &rhs) {
  if (rhs.full()) {
    ost << "no-merge";
  } else {
    ost << rhs.first();
  }
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const std::vector<Sett> &setts) {
  ost << '(';
  for (const auto &x : setts) {
    x.append(ost);
  }
  ost << ')';
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const DisjointSetts &setts) {
  ost << setts.get();
  return ost;
}

DisjointSetts Sett::fill(const Sett &scaffold, const Sett &ink) {

  const auto nOnScaffold = scaffold.n(0, scaffold.period());
  const auto scm      = smallestCommonMultiple_i64(nOnScaffold, ink.period());
  const auto nRepls   = scm / nOnScaffold;
  int64_t scaffoldUpp = scaffold.period() * nRepls;

  auto filled = canonicalized(fillRecurse(scaffold,
                                          ink,
                                          0, // depth
                                          scaffoldUpp));
  if (log().shouldLogTrace()) {
    std::ostringstream oss;
    oss << "In Sett::fill(" << scaffold << ", " << ink
        << "), filled=" << filled;
    log().trace(oss.str());
  }

  return filled;
}

DisjointSetts Sett::sample(const Sett &x, const Sett &filter) {

  if (x.equivalent(filter)) {
    return {Sett::createAlwaysOn()};
  }

  const auto uppCorrect = filter.n(0, x.smallestCommonMultiple(filter));

  auto sampled = sampleRecurse(x,
                               filter,
                               0,   // depth
                               0LL, // lowCorrect
                               uppCorrect);

  for (auto &s : sampled) {
    if (s.period() % uppCorrect != 0) {
      s = s.adjustedPrepend({uppCorrect, 0, 0});
    }
  }

  sampled = canonicalized(sampled);

  if (log().shouldLogTrace()) {
    std::ostringstream oss;
    oss << "In Sett::sample(" << x << ", " << filter
        << "), sampled=" << sampled;
    log().trace(oss.str());
  }
  return sampled;
}

//

OptionalSett1 Sett::mergeA(const Sett &lhs, const Sett &rhs) {

  // Example:
  //
  // ....1...11...11...11..........1...11...11...11........  a
  // 1......................11........................11...  b
  // 012345678901234567890123456789012345678901234567890123
  // 1...1...11...11...11...11.....1...11...11...11...11...  merged.
  //
  // a = (...)(16, 10, 4)(2, 3, 4)(...)
  // b = (...)(2, 24, 23)(...)
  //

  const auto depthOrdered = getDepthOrdered(lhs, rhs);
  const auto &shallow     = *std::get<0>(depthOrdered);
  const auto &deep        = *std::get<1>(depthOrdered);
  if (shallow.recursiveDepth_u64() + 1 == deep.recursiveDepth_u64()) {

    // find first depth where the stripe is different:
    const auto depth0     = shallow.depthWhereFirstDifference(deep);
    const auto depth0_u64 = static_cast<uint64_t>(depth0);
    if (depth0_u64 < shallow.recursiveDepth_u64() &&
        shallow.fromDepth(depth0 + 1).getStripes() ==
            deep.fromDepth(depth0 + 2).getStripes()) {

      const auto &s0 = shallow.atDepth(depth0_u64);
      const auto &d0 = deep.atDepth(depth0_u64);
      if (s0.period() == d0.period()) {

        const auto &d1 = deep.atDepth(depth0_u64 + 1);
        if (s0.on() == d1.on()) {
          const auto delta = d0.phase() + d1.phase() - s0.phase();
          if ((delta % d1.period() == 0)) {
            auto extOn  = d0.on() + d1.period();
            auto extOff = d0.period() - extOn;
            if (extOff >= 0) {

              // try extensions. There are more tests we could use to
              // narrow in on the equivalance, but to save time,
              // code, and potential bugs, just checking directly.

              // try extending forward and backward:
              for (int64_t extensionPhase :
                   {d0.phase(), d0.phase() - d1.period()}) {

                Stripe ext1{extOn, extOff, extensionPhase};
                Sett extended{{{ext1, d1}}};
                std::vector<Sett> targ{{{{d0, d1}}}, {{{s0}}}};

                if (extended.equivalent(DisjointSetts(std::move(targ)))) {
                  auto merged                = deep;
                  merged.stripes[depth0_u64] = ext1;
                  return {{merged}};
                }
              }
            }
          }
        }
      }
    }
  }

  return OptionalSett1::None();
}

OptionalSett2 Sett::transferA(const Sett &lhs, const Sett &rhs) {
  const auto depthOrdered = getDepthOrdered(lhs, rhs);
  const auto &shallow     = *std::get<0>(depthOrdered);
  const auto &deep        = *std::get<1>(depthOrdered);
  if (shallow.recursiveDepth_u64() + 1 == deep.recursiveDepth_u64()) {
    const auto depth0     = shallow.depthWhereFirstDifference(deep);
    const auto depth0_u64 = static_cast<uint64_t>(depth0);
    if (depth0_u64 < shallow.recursiveDepth_u64()) {

      // try and split "deep" into 2 of size depth0, like a reverse mergeC.
      const auto &d0 = deep.atDepth(depth0_u64 + 0);
      const auto &d1 = deep.atDepth(depth0_u64 + 1);
      //...1111111111......  d0
      //    111..111.......  d1
      const auto s0 = d1.firstStartNotBefore(0);
      const auto s1 = s0 + d1.period();
      const auto e1 = s1 + d1.on();
      const auto s2 = s1 + d1.period();
      if (s0 - d1.off() <= 0 && e1 <= d0.on() && s2 >= d0.on()) {
        std::vector<Stripe> left;
        left.reserve(shallow.recursiveDepth_u64());
        const auto &deepStripes = deep.getStripes();
        left.insert(left.end(),
                    deepStripes.cbegin(),
                    std::next(deepStripes.cbegin(), depth0));

        const Stripe leftSub{
            d1.on(), d0.period() - d1.on(), d0.phase() + d1.phase()};

        const Stripe rightSub{d1.on(),
                              d0.period() - d1.on(),
                              d0.phase() + d1.phase() + d1.period()};

        left.push_back(leftSub);
        left.insert(left.end(),
                    std::next(deepStripes.cbegin(), depth0 + 2),
                    deepStripes.cend());
        const Sett leftSett{left};

        auto right        = left;
        right[depth0_u64] = rightSub;

        const Sett rightSett{right};

        auto subMerged = mergeB(shallow, leftSett);
        if (subMerged.full()) {
          return {{subMerged.first(), rightSett}};
        }
        subMerged = mergeB(shallow, rightSett);
        if (subMerged.full()) {
          return {{subMerged.first(), leftSett}};
        }
      }
    }
  }
  return OptionalSett2::None();
}

void Sett::confirmDisjoint(const std::vector<Sett> &rhs) {
  if (!disjoint(rhs)) {
    std::ostringstream oss;
    oss << "Failure in confirmDisjoint, the Setts " << rhs
        << " are not all disjoint.";
    throw error(oss.str());
  }
}

OptionalSett1 Sett::mergeB(const Sett &lhs, const Sett &rhs) {
  //
  // Attempt to merge (...)(sett0_a)(sett1_a)(...)
  //                  (...)(sett0_b)(sett1_b)(...)
  //
  // by concatenating at depth sett0 : sett0_a and sett0_b.
  //
  // example:
  // ..1111.....1111..... (...)(4,5,2)
  // .1........1........1 (...)(1,8,1)
  //
  // becomes:
  // .11111....11111....1 (...)(5,4,1)

  // same depth, but not depth-0:
  if (lhs.recursiveDepth_u64() == rhs.recursiveDepth_u64() &&
      lhs.recursiveDepth_u64() > 0) {
    const auto depth0     = lhs.depthWhereFirstDifference(rhs);
    const auto depth0_u64 = static_cast<uint64_t>(depth0);
    const auto totDepth   = lhs.recursiveDepth_u64();

    // not identical, but same in range after sett1_a (sett1_b):
    if (depth0 < lhs.recursiveDepth() &&
        sameInDepthRange(lhs, rhs, depth0_u64 + 2, totDepth)) {
      const auto &lhs0 = lhs.atDepth(depth0_u64);
      const auto &rhs0 = rhs.atDepth(depth0_u64);
      if (lhs0.period() == rhs0.period()) {
        const auto ordered = getPhaseOrdered(lhs0, rhs0);
        const auto &early  = *std::get<0>(ordered);
        const auto &late   = *std::get<1>(ordered);
        if (early.phase() + early.on() == late.phase()) {
          auto sumTailOn = early.on() + late.on();
          auto newOff    = early.period() - sumTailOn;
          if (newOff >= 0) {
            auto merged           = lhs;
            merged.stripes.back() = {
                sumTailOn, early.period() - sumTailOn, early.phase()};

            // Note that this final check for equivalence picks up at least
            // 2 edge cases:
            // - intersecting
            // - at one depth below depth0, the stripes are not aligned
            if (disjoint({lhs, rhs})) {
              if (merged.equivalent(DisjointSetts{{lhs, rhs}})) {
                return {{merged}};
              }
            }
          }
        }
      }
    }
  }
  return OptionalSett1::None();
}

OptionalSett1 Sett::mergeC(const Sett &lhs, const Sett &rhs) {

  // 1.....1.....1..... (...)(1,5,0)
  // ..1......1.....1.. (...)(1,5,2)
  // (...)(3,3,0)(1,1,0)
  //
  if (lhs.recursiveDepth_u64() == rhs.recursiveDepth_u64()) {
    const auto depth0     = lhs.depthWhereFirstDifference(rhs);
    const auto depth0_u64 = static_cast<uint64_t>(depth0);
    const auto totDepth   = lhs.recursiveDepth_u64();
    if (depth0_u64 != totDepth &&
        sameInDepthRange(lhs, rhs, depth0_u64 + 1, totDepth)) {
      const auto &lhs0 = lhs.atDepth(depth0_u64);
      const auto &rhs0 = rhs.atDepth(depth0_u64);
      if (lhs0.on() == rhs0.on() && lhs0.off() == rhs0.off()) {
        const auto ordered = getPhaseOrdered(lhs0, rhs0);
        const auto &early  = *std::get<0>(ordered);
        const auto &late   = *std::get<1>(ordered);
        if (early.phase() + early.on() <= late.phase()) {

          const auto &lhsStripes = lhs.getStripes();

          std::vector<Stripe> mergedStripes;
          mergedStripes.reserve(totDepth + 1);
          mergedStripes.insert(mergedStripes.end(),
                               lhsStripes.cbegin(),
                               std::next(lhsStripes.cbegin(), depth0));

          auto glueOn = late.on() + late.phase() - early.phase();
          Stripe glue{glueOn, lhs0.period() - glueOn, early.phase()};
          mergedStripes.push_back(glue);
          mergedStripes.push_back(
              {lhs0.on(), late.phase() - early.phase() - early.on(), 0});
          mergedStripes.insert(mergedStripes.end(),
                               std::next(lhsStripes.cbegin(), depth0 + 1),
                               lhsStripes.cend());
          Sett merged(std::move(mergedStripes));
          if (disjoint({lhs, rhs}) &&
              merged.equivalent(DisjointSetts({lhs, rhs}))) {
            return {{merged}};
          }
        }
      }
    }
  }

  return OptionalSett1::None();
}

DisjointSetts Sett::canonicalized(const DisjointSetts &setts) {

  std::vector<Sett> canon0;
  canon0.reserve(setts.size());
  for (const auto &x : setts.get()) {
    auto sett = Sett{x.getStripes(), true};
    if (!sett.alwaysOff()) {
      canon0.push_back(sett);
    }
  }

  // some simple merging.
  std::vector<Sett> canon1;

  bool improvement = true;
  while (improvement) {
    improvement = false;
    std::swap(canon1, canon0);
    canon0.clear();

    for (const auto &sett : canon1) {
      if (sett.alwaysOff()) {
      } else if (canon0.empty()) {
        canon0.push_back(sett);
      } else {
        auto merged = merge(canon0.back(), sett);
        improvement |= merged.full();
        if (merged.full()) {
          canon0.back() = merged.first();
        } else {
          auto transferred = transfer(canon0.back(), sett);
          improvement |= transferred.full();
          if (transferred.full()) {
            canon0.back() = transferred.get<0>();
            canon0.push_back(transferred.get<1>());
          } else {
            canon0.push_back(sett);
          }
        }
      }
    }
  }

  canon1.clear();
  for (const auto &x : canon0) {
    auto sett = Sett{x.getStripes(), true};
    if (!sett.alwaysOff()) {
      canon1.push_back(sett);
    }
  }

  return DisjointSetts(std::move(canon1));
}

OptionalSett2 Sett::transfer(const Sett &lhs, const Sett &rhs) {
  return transferA(lhs, rhs);
}

OptionalSet<1, Sett> Sett::merge(const Sett &lhs, const Sett &rhs) {
  auto merged = mergeA(lhs, rhs);
  if (merged.full()) {
    return merged;
  }

  merged = mergeB(lhs, rhs);
  if (merged.full()) {
    return merged;
  }

  return mergeC(lhs, rhs);
}

int Sett::depthWhereFirstDifference(const Sett &rhs) const {
  const auto minDepth =
      std::min(recursiveDepth_u64(), rhs.recursiveDepth_u64());
  for (uint64_t i = 0; i < minDepth; ++i) {
    if (atDepth(i) != rhs.atDepth(i)) {
      return static_cast<int>(i);
    }
  }
  return static_cast<int>(minDepth);
}

DisjointSetts Sett::fillRecurse(const Sett &scaffold,
                                const Sett &ink,
                                int depth,
                                int64_t scaffoldUpp) {

  const auto nScaffoldOnInRequiredRange = scaffold.n(0, scaffoldUpp);
  const auto nInkOnInRequiredRange = ink.n(0, nScaffoldOnInRequiredRange);

  // the base (termination) cases of the recursion:

  // scaffold always off in [0, scaffoldUpp)
  //    ==> can assume scaffold is always off
  //    ==> return always off.
  if (nScaffoldOnInRequiredRange == 0) {
    return {};
  }

  // scaffold always on in [0, scaffoldUpp)
  //    ==> can assume scaffold is always on
  //    ==> return ink.
  if (nScaffoldOnInRequiredRange == scaffoldUpp) {
    return {ink};
  }

  // ink always off in [0, nScaffoldOnInRequiredRange)
  //    ==> can assume ink is always off
  //    ==> return always off.
  if (nInkOnInRequiredRange == 0) {
    return {};
  }

  // ink always on in [0, nScaffoldOnInRequiredRange)
  //    ==> can assume ink is always on
  //    ==> return scaffold.
  if (nInkOnInRequiredRange == nScaffoldOnInRequiredRange) {
    return {scaffold};
  }

  // None of the 4 base cases stuck, so we are guaranteed that ink and
  // scaffold are both have Stripes:
  const auto &ink0 = ink.atDepth(0);
  const auto &sca0 = scaffold.atDepth(0);

  const auto nOnScaffold = scaffold.n(0, sca0.period());
  const auto inkOffset0  = scaffold.n(0, sca0.phase());
  // Example of above, if scaffold is {{5,3,5}, {1,1,0}}:
  // 11...11111...11111...11111..
  // .1...1.1.1...1.1.1...1.1.1..
  // then nOnScaffold = 3, and inkOffset0 = 1.

  // recursive case a: reduce scaffold depth by 1.
  if (ink0.period() == nOnScaffold) {
    auto offInk = ink;
    offInk.changeFirstStripe(
        {ink0.on(), ink0.off(), ink0.phase() - inkOffset0});

    auto subFilled = fillRecurse(scaffold.fromDepth(1),
                                 offInk,
                                 depth + 1, // new depth
                                 sca0.on()  // new scaffoldUpp
    );
    for (auto &x : subFilled) {
      x.prependStripes({sca0});
    }
    return subFilled;
  }

  const auto scm    = smallestCommonMultiple_i64(nOnScaffold, ink0.period());
  const auto nRepls = scm / nOnScaffold;

  // recursive case b: reduce ink depth by 1.
  if (ink0.period() % nOnScaffold == 0) {

    // Example:
    // ink = {{7,5,6}, {1,1,0}}
    // 1.....1.1.1.1.....1.1.1.1.....1.1.1.1.....
    // scaffold = {3,1,2}
    // 1.111.111.111.111.111.111.
    //
    // Expected output:
    // 1 ... ..1 .1. 1.1 ... ..1 .1. 1.1 ...  ink
    // 1.111.111.111.111.111.111.111.111.111. scaffold
    // 1.......1..1..1.1.......1..1..1.1.....
    // 0 123 456 789 012 3  // the n'th on in scaffold
    // 0123456789012345678  // the absolute position.
    //         1111111111      1111111111  // the super-stripe.
    //
    // upFactor = 12/3 = 4
    // superPeriod = 4*4 = 16
    // superStart = scaffold.getOn(6) = 8
    // superEnd = scaffold.getOn(13) = 18
    // superOn = 10
    // superOff = 6.
    // recursiveCall ({{3,1,2}},{{1,1,0}})
    const auto upFactor    = ink0.period() / nOnScaffold;
    const auto superPeriod = upFactor * sca0.period();
    const auto superStart  = scaffold.getOn(ink0.phase());
    const auto superEnd    = scaffold.getOn(ink0.phase() + ink0.on());
    const auto superOn     = superEnd - superStart;
    const auto superOff    = superPeriod - superOn;
    const Stripe superStripe{superOn, superOff, superStart};
    if (superStripe.allOff(0, scaffoldUpp)) {
      return {};
    }
    auto shiftedScaffold = scaffold.phaseShifted(-1 * superStart);
    auto subFilled =
        fillRecurse(shiftedScaffold, ink.fromDepth(1), depth + 1, superOn);
    for (auto &x : subFilled) {
      x.prependStripes({superStripe});
    }
    return subFilled;
  }

  // recursive case c: increase ink depth by 1, but ensure case a is next, so
  // effectively trasfer a stripe from ink to scaffold.
  if (nOnScaffold != ink0.period()) {

    // Example, scaffold = {{6,2,0}, {3,1,-1}}.
    // 11.111..11.111..11.111..11.111..11.111..
    // nOnScaffold = 5, inkOffset0 = 0.
    //
    // Example, ink = {2,2,0}
    // 11..11..11..11
    //
    //                                    |
    // x1.111.. 11.111.. 11.111.. 11.111..|11.111.. 11.111.. 11.111.. scaffold
    // 11 ..1   1. .11   .. 11.   .1 1..  |11 ..1   1. .11   .. 11.   ink
    //                                    |
    //

    std::vector<Sett> filled;

    auto sparseScaffold = scaffold;
    for (int64_t repl = 0; repl < nRepls; ++repl) {
      sparseScaffold.changeFirstStripe({sca0.on(),
                                        nRepls * sca0.period() - sca0.on(),
                                        sca0.phase() + repl * sca0.period()});

      auto sparseInk = ink;

      sparseInk.changeFirstStripe(
          {ink0.on(),
           ink0.off(),
           ink0.phase() - inkOffset0 - nOnScaffold * repl});

      sparseInk.prependStripes(
          {{nOnScaffold,
            0,
            +1 * sparseScaffold.n(0, sparseScaffold.atDepth(0).phase())}});

      const auto partsFilled =
          fillRecurse(sparseScaffold, sparseInk, depth + 1, scaffoldUpp);
      filled.insert(filled.end(), partsFilled.cbegin(), partsFilled.cend());
    }
    return DisjointSetts(std::move(filled));
  }

  throw error("Internal logic error, should never reach this point in fill");
}

DisjointSetts Sett::fillWith(const Sett &ink) const {
  return fill(*this, ink);
}

Sett Sett::phaseShifted(int64_t deltaPhase0) const {
  if (!hasStripes()) {
    return {{}};
  }

  auto shifted = getStripes();
  auto s0      = shifted[0];
  shifted[0]   = {s0.on(), s0.off(), s0.phase() + deltaPhase0};
  return Sett(shifted);
}

void Sett::shiftPhase(int64_t deltaPhase0) {

  if (hasStripes()) {
    auto shifted = getStripes();
    auto s0      = atDepth(0);
    stripes[0]   = {s0.on(), s0.off(), s0.phase() + deltaPhase0};
  }
}

DisjointSetts Sett::getNonCrossingB(int64_t p) const {

  if (!hasStripes()) {
    return {createAlwaysOn()};
  }

  const auto &s0 = atDepth(0);
  auto x0        = s0.phase();
  auto x1        = x0 + s0.on();

  if (x0 / p == x1 / p) {
    return {*this};
  }

  auto z1 = p * (x0 / p + (x0 % p != 0));
  auto z2 = p * (x1 / p);

  // 0      p      2p     3p     4p
  // |      |      |      |      |
  //     X            X
  //     ------------
  //       z1     z2

  auto onLeft  = z1 - x0;
  auto onMid   = z2 - z1;
  auto onRight = x1 - z2;

  std::vector<Sett> nonCrossings;
  auto appendNonCrossing = [&nonCrossings, s0, this](int64_t on,
                                                     int64_t dPhase) {
    if (on != 0) {
      Stripe pref{on, s0.period() - on, s0.phase() + dPhase};
      auto sett = fromDepth(1);
      sett      = sett.phaseShifted(-dPhase);
      sett.prependStripes({pref});
      nonCrossings.push_back(sett);
    }
  };

  appendNonCrossing(onLeft, 0);
  appendNonCrossing(onMid, onLeft);
  appendNonCrossing(onRight, onLeft + onMid);

  return DisjointSetts(std::move(nonCrossings));
}

DisjointSetts Sett::getNonCrossingA(int64_t l0, int64_t u0) const {

  //
  //
  //        l0                  l1
  //        |                   |
  //     ..11.1..11.1..11.1..11.1..11.1..11
  //       s     s     s     s   (starts of first stripe of this)
  // return:
  //        xxxx=============+++
  //        1.1.................
  //        .....11.1..11.1.....
  //        .................11.
  //
  //
  //

  if (!hasStripes()) {
    return {createAlwaysOn()};
  }

  const auto &s0 = atDepth(0);

  if (s0.allOff(l0, u0)) {
    return {};
  }

  const auto start0 = s0.firstStartNotBefore(l0);
  const auto nFull =
      std::max<int64_t>(0LL, (u0 - start0 + s0.off()) / s0.period());

  const auto z0 = l0;
  const auto z1 = start0;
  const auto z2 = std::min(u0, start0 + nFull * s0.period());
  const auto z3 = u0;

  std::vector<Sett> nonCrossings;

  const auto superPeriod = u0 - l0;
  auto appendNonCrossing =
      [&nonCrossings, this, z0, superPeriod](int64_t za, int64_t zb) {
        if (zb > za) {
          const auto on  = zb - za;
          const auto off = superPeriod - on;
          Stripe super{on, off, za - z0};
          auto sett = phaseShifted(-za);
          sett.prependStripes({super});
          sett = Sett(sett.getStripes());
          if (!sett.alwaysOff()) {
            nonCrossings.push_back(sett);
          }
        }
      };

  appendNonCrossing(z0, z1);
  appendNonCrossing(z1, z2);
  appendNonCrossing(z2, z3);

  return DisjointSetts(std::move(nonCrossings));
}

/**
 * Refactor this Sett into a partition of Setts, where the
 * first Stripe in each Sett in the partition has p as a factor of its
 * period. Furthermore, each outermost Stripe will not be
 * split-and-hidden.
 * (read example below for details of what split-and-hidden means)
 *
 * Example 1:
 *
 * If this is (4,3,5) and p is 100:
 * 11...1111...1111...1111...1111...1111..(...),
 *
 * there are 3 ways of dividing this to have first Strips's periods all
 * multiples of p:
 *
 * A) break it into 700 Setts, of on=4 and off=696 and depth 1:
 *    (4,696,5), (4,696,12) .... etc.
 *
 * B) break it into 7 Setts, with outermost Stripe of
 *    on=100, off=600, phase=varying.
 *      (100,600,0)(4,3,5) (100,600,100)(4,3,3) ... etc.
 *
 * C) like B, but further divide any Setts which have hidden-splits.
 *     For example, (100,600,0)(4,3,5) becomes (2,698) and (95,605,5)(4,3,0)
 *
 * In terms of total Setts, B < C < A, so A seems like a good choice.
 * However, with solution B, the outermost Stripe is split-and-hidden:
 * (4,3,5) has incomplete periods in (100,600,0). So C is the best
 * permissible solution in this case. Avoiding hidden-splits is a desirable
 * feature for retaining rectangular regions in reshapes (jn:
 * reconsider this statement: is it really worth the increased shattering with
 * the non-strider approach?)
 *
 * Example 2:
 *
 * 1..1..1..1..1..1..1..1..1..1..1..1..1..1..1..1..(...)
 * 0                   20                  40
 * Where this is ((1,2,0)) and p is 20, best (i.e. fewest Setts in
 * solution) permissible solution is,
 * (((20, 40, 0),  (1,2,0)),
 *  ((20, 40, 20), (1,2,2)),
 *  ((20, 40, 40), (1,2,1))).
 *
 *
 * Example 3:
 *
 * .11...11...11...11...11...11...11.(...)
 * 0         10        20        30
 * Where this is ((2,3,1)) and p is 10, best permissible solution is,
 * (((2,3,1))).
 *
 *
 * Example 4:
 *
 *  Where this is ((12, 3, 1)) and p is 20
 * .111111111111...111111111111...111111111111
 * 0              15   20        30        40
 *
 * A non-permissible solution is:
 * (((20, 40, 0),   (12, 3, 1))
 *  ((20, 40, 20),  (12, 3, 1-20)
 *  ((20, 40, 40),  (12, 3, 1-40))).
 *
 * It is not permissible as there are hidden-splits. A permissible solution
 * is further divides these.
 *
 *
 * Example 5:
 *
 * Where this is ((12, 3, 1)) and p is 10, a permissible solution is
 * (((12, 18, 1 )), ((12, 18, 16))).
 *
 *
 * Example 6:
 *
 * Where this is ((12, 3, 1)) and p is 5, a permissible solution is (((12,
 * 3, 1))).
 *
 *
 *  Note the different optimal approaches when the first Stripe has period
 * smaller than p, and larger than p. The approach taken in each case
 *  minimizes the number of Setts returned in the vector.
 *
 *
 * \param p the factor that every Sett in the returned partition
 *          must have as a factor of its outermost Stripe's period.
 *
 * \param upper It is only required that the intersection of the returned set
 *              in the range [0,upper) is correct.
 *
 * */
DisjointSetts Sett::getNonCrossingPeriodic(int64_t p, int64_t upper) const {

  if (!hasStripes()) {
    return {createAlwaysOn()};
  }

  const auto &s0 = atDepth(0);

  std::vector<Sett> soln;
  auto scm = smallestCommonMultiple_i64(s0.period(), p);
  if (p > s0.period()) {
    const auto repl = scm / p;

    // iterate repl times, unless upper dictates you can stop early.
    // A stripe with phase=i*p and on=p is prepended (see below), os
    // if i*p < upper, then redundant.
    const auto iterCount = std::min<int64_t>(1LL + upper / p, repl);
    for (int64_t i = 0; i < iterCount; ++i) {
      for (auto sett : getNonCrossingA(i * p, (i + 1) * p)) {
        sett.prependStripes({{p, scm - p, i * p}});
        soln.push_back(sett);
      }
    }
  } else {

    // xxx....xxxxxxxxxxxxxxxx....xxxx
    // 0123456789012345678901234567890
    // |    |    |    |    |    |    |
    //
    const auto repl = scm / s0.period();

    // iterate repl times, unless upper dictates you can stop early.
    // Check if [i*period + phase, (i+1)*period + phase) intersects [0,
    // upper).
    const auto iterStart = -1LL;
    const auto iterEnd =
        std::min(1 + (upper - s0.phase()) / s0.period(), repl - 1);

    for (int64_t i = iterStart; i < iterEnd; ++i) {
      const Stripe prefix{
          s0.on(), scm - s0.on(), s0.phase() + i * s0.period()};
      auto x = fromDepth(1);
      x.prependStripes({prefix});
      for (auto y : x.getNonCrossingB(p)) {
        soln.push_back(y);
      }
    }
  }

  return DisjointSetts(std::move(soln));
}

std::vector<std::array<Sett, 2>> Sett::unflatten(int64_t p) const {
  auto scm = smallestCommonMultiple_i64(p, period()) / p;
  return unflattenRecurse(p, scm, 0);
}

std::vector<std::array<Sett, 2>>
Sett::unflattenRecurse(int64_t p, int64_t h1, int depth) const {

  if (!hasStripes()) {
    return {{Sett::createAlwaysOn(), Sett::createAlwaysOn()}};
  }

  std::vector<std::array<Sett, 2>> soln;

  const auto upperRequired  = p * h1;
  auto nonCrossingPeriodics = getNonCrossingPeriodic(p, upperRequired);

  for (auto sett_2 : nonCrossingPeriodics) {
    const auto &s0 = sett_2.atDepth(0);

    // while getNonCrossingPeriodic partially filtered out the Setts out of
    // the required range, it did catch all of them (due to nesting - it only
    // considered depth 0).
    if (sett_2.n(0, p * h1) == 0) {
      continue;
    }

    // first case: p greater than or equal to Sett's period, so
    //
    // .111..111..111..111..111..111..111.  (1-d)
    // [              [              [
    //
    // becomes
    //
    // .111..111..111.
    // .111..111..111. (2-d).
    // .111..111..111.
    //
    if (p % s0.period() == 0) {
      soln.push_back({Sett::createAlwaysOn(), sett_2});
    }

    // second case : p less than Sett's period
    else if (s0.period() % p == 0) {
      const auto on0    = s0.on();
      const auto phase0 = s0.phase();

      // factorize out the largest Stripe, removing it from the "flat" term
      const auto superPhase  = phase0 / p;
      const auto superPeriod = s0.period() / p;
      const auto superOn     = on0 / p + (on0 % p != 0);
      Stripe superStripe{superOn, superPeriod - superOn, superPhase};

      // case of no recursive call
      //
      // Example (2,10,0) with p=4.
      //
      // superPhase = 2, superPeriod = 3, superOn = 1.
      //
      // ..11..........11..........11..
      // [   [   [   [   [   [   [   [
      //
      //           ..11
      //           ....
      //           ....
      // becomes   ..11
      //           ....
      //           ....
      //           ..11
      //
      if (on0 <= p) {
        sett_2.changeFirstStripe({on0, p - on0, phase0 % p});
        soln.push_back({Sett{{superStripe}}, sett_2});
      }

      // case of recursive call
      else {
        // some logic checking:
        if (on0 % p != 0) {
          std::ostringstream oss;
          oss << "In unFlatten, with on0 (" << on0 << ") not divisible by p ("
              << p << "). This should have been guaranteed by the "
              << "earlier call to getNonCrossingPeriodic, "
              << "something has gone wrong. ";
          throw error(oss.str());
        }
        auto fromDepth1 = sett_2.fromDepth(1);
        auto subunflatteneds =
            fromDepth1.unflattenRecurse(p, superOn, depth + 1);
        for (auto subun : subunflatteneds) {
          auto pref = std::get<0>(subun);
          auto suff = std::get<1>(subun);
          pref.prependStripes({superStripe});
          soln.push_back({pref, suff});
        }
      }
    } else {
      std::ostringstream oss;
      oss << "The period of s0 (" << s0.period()
          << ") is not divisible by p (" << p << "). "
          << "This is unexpected, the call to getNonCrossingPeriodic(" << p
          << ") should have ensured that it is. "
          << "This point should be unreachable. ";
      throw error(oss.str());
    }
  }

  return soln;
}

std::vector<Sett>
Sett::scaledConcat(const std::vector<std::array<Sett, 2>> &unflat,
                   int64_t p) {
  std::vector<Sett> flatteneds;
  for (uint64_t i = 0; i < unflat.size(); ++i) {
    const auto &u0 = std::get<0>(unflat[i]);
    const auto &u1 = std::get<1>(unflat[i]);
    std::vector<Stripe> reflattened;
    reflattened.reserve(u0.recursiveDepth_u64() + u1.recursiveDepth_u64());
    for (const auto &s : u0.getStripes()) {
      reflattened.push_back(s.getScaled(p));
    }
    for (const auto &s : u1.getStripes()) {
      reflattened.push_back(s);
    }
    flatteneds.push_back({reflattened});
  }
  return flatteneds;
}

DisjointSetts Sett::getComplement() const {

  if (!hasStripes()) {
    return {};
  }
  if (alwaysOff()) {
    return createAlwaysOn();
  }

  std::vector<Sett> complements;
  complements.push_back(Sett({atDepth(0).getComplement()}));
  auto subComplement = fromDepth(1).getComplement();
  for (auto &sub : subComplement) {
    sub.prependStripes({atDepth(0)});
    complements.push_back(sub);
  }

  return DisjointSetts(complements);
}

DisjointSetts Sett::subtract(const Sett &rhs) const {
  const auto rhsCompl = rhs.getComplement();
  std::vector<Sett> diff;
  for (const auto &rhsComplElm : rhsCompl.get()) {
    const auto inter = intersect(rhsComplElm);
    diff.insert(diff.end(), inter.cbegin(), inter.cend());
  }
  return DisjointSetts(diff);
}

} // namespace nest
} // namespace memory
} // namespace poprithms
