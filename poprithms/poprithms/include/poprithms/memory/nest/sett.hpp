// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_NEST_SETT_HPP
#define POPRITHMS_MEMORY_NEST_SETT_HPP

#include <vector>

#include <poprithms/memory/nest/optionalset.hpp>
#include <poprithms/memory/nest/stripe.hpp>

namespace poprithms {
namespace memory {
namespace nest {

// Online definition of "sett":
//  >  The particular pattern of stripes in a tartan
//
// Online etymology of "sett":
//  >  Middle English: variant of set, the spelling with -tt
//  >  prevailing in technical senses.

class Sett;
using Setts = std::vector<Sett>;
class DisjointSetts;
using OptionalSett1 = OptionalSet<1, Sett>;
using OptionalSett2 = OptionalSet<2, Sett>;

int64_t smallestCommonMultiple_i64(int64_t a, int64_t b);

/**
 * Nested Stripes, used to represent more complicated periodic on-off
 * patterns than what a single Stripe can.
 *
 * A Sett is completely defined by its lone class member,
 * std::vector<Stripe> stripes;
 *
 * Some examples of the patterns defined by stripes:
 *
 * stripes = {{on=4,off=2,phase=0}}     A single Stripe (no nesting, so this
 * 1111..1111..1111.. etc               is just the pattern described by a
 * .                                    single Stripe).
 *
 * stripes = {{on=6,off=2,phase=1}}     A single Stripe, again.
 * .111111..111111..111111.. etc
 *
 * stripes = {{on=2,off=1,phase=0}}     A single Stripe, again.
 * 11.11.11.11.11.11.11.11. etc
 *
 * stripes = {{on=6,off=2,phase=4}}     Another single Stripe.
 * 11..111111..111111..111111.. etc
 *
 * stripes = {}                         No Stripes, implies always on.
 * 111111111111111111111111111  etc
 *
 * stripes = {{on=1,off=0,phase=0}}     Another representation of
 * 111111111111111111111111111  etc     an always on Sett.
 *
 * stripes = {{on=5,off=3,phase=1},     2 nested Stripes. The second Stripe is
 *            {on=2,off=1,phase=0}.     nested within the first, and its
 * .11.11...11.11...11.11...11.11       phase is relative to the first's.
 *
 * Note that nesting of Stripes is not intersection of Stripes. The second
 * Stripe above is "restarted" every time the first (nesting) Stripe
 * transitions from off to on:
 *
 * .11111...11111...11111...11111...11111  the first / outermost Stripe above
 *  11.11.11.11                            the inner Stripe, nesting 1
 *  =====
 *       11.11.11.1                        nesting 2
 *          =====
 *               11.11.11.11.              nesting 3
 *                  =====
 *  11.11   11.11   11.11   11.11          Sett described by the 2 Stripes.
 *
 * As another example
 * {{on=1,off=2,phase=0},{on=1,off=0,phase=0}} is equivalent to
 * {{on=1,off=2,phase=0}}, so the always on inner Stripe has no effect.
 *
 * However, an always on outer Stripe cannot be ignored:
 * {{on=3,off=0,phase=0}, {on=1,off=1,phase=0}} looks like
 * 1.11.11.11.11.11.11.11.11.1
 * ===---===---===---===---===.
 *
 * stripes = {{on=6,off=2,phase=4},
 *            {on=2,off=1,phase=0}     Another example of 2 nested Stripes.
 * 1...11.11...11.11...11.11... etc
 *
 * stripes = {{on=5,off=2,phase=0},
 *            {on=1,off=1,phase=0}     2 nested Stripes, again. Note how the
 *  1.1.1..1.1.1..1.1.1..1.1.1. etc    nested Stripe's phase is relative to
 * .                                   the nesting Stripe's phase.
 *
 * stripes = {{on=8,off=8,phase=-2},   3 nested Stripes
 *            {on=3,off=3,phase=1},
 *            {on=1,off=1,phase=0}.
 *  111111........11111111........11111111........11111111 outermost
 *  11...1        .111...1        .111...1        .111...1 middle
 *  .1   1         1.1   1         1.1   1         1.1   1 innermost
 *  .1...1.........1.1...1.........1.1...1.........1.1...1 final pattern.
 *
 * More examples can be seen in the test directory.
 *
 * Sett is a useful abstraction for representing regions of Tensors
 * which are sliced, reshaped, etc. They serve a similar purpose in
 * this project to Poplar's Tensor expression trees, and can be thought of as
 * an extension to the boost::intervals used in there. Using
 * a generalised interval such as the Sett means that less interval
 * "shattering" happens, resulting in speed-ups for the higher level
 * Graph transformations encountered at the ML frameworks level.
 *
 * Most of the methods of the class are standard set operations, where
 * the set is the '1's described by class member variable, stripes.
 * */
class Sett {

public:
  Sett()  = delete;
  ~Sett() = default;

  Sett(const Sett &) = default;
  Sett(Sett &&)      = default;

  Sett &operator=(const Sett &) = default;
  Sett &operator=(Sett &&) = default;

  /**
   * Construct a Sett from a sequence of Stripes
   *
   * \param stripes_ Nested Stripes, with the innermost (most nested) Stripe
   *                 being at the back of the vector and the outermost Stripe
   *                 appearing first.
   * */
  Sett(const std::vector<Stripe> &stripes_) : Sett(stripes_, true) {}

  /**
   * Construct a Sett from a nested sequence of Stripes, and optionally
   * perform canonicalization to simplify stripes.
   *
   * \param stripes_ Nested Stripes, with the most nested being the at the
   *                 back of the vector and the outermost Stripe appearing
   *                 first.
   *
   * \param canonicalize If true, try to simplify the Sett's stripes to an
   *                     equivalent representation.
   *
   * As an example of canonicalization, {{1,1,0}, {2,2,0}} might become the
   * equivalent but simpler {{1,1,0}}.
   * */
  Sett(const std::vector<Stripe> &stripes_, bool canonicalize);

  /**
   * Factory function to return an always '1' (always on) Sett.
   */
  static Sett createAlwaysOn() { return {{}, false}; }

  /**
   * Factory function to return an always '0' (always off) Sett.
   */
  static Sett createAlwaysOff() {
    return {{Stripe{/*on=*/0, /*off=*/1, /*phase=*/0}},
            /*canonicalize=*/false};
  }

  /**
   * \return The smallest period over which this and rhs repeat. This is just
   *         the smallest common multiple of the periods of the outermost
   *         Stripes (if they not stripeless).
   * */
  int64_t smallestCommonMultiple(const Sett &rhs) const;

  /**
   * \return The smallest period over which a set of Setts repeat. This is
   *         just the smallest common multiple of the periods of the outermost
   *         Stripes (if not stripeless).
   *
   * \param setts The set to compute the smallest common period for.
   * */
  static int64_t smallestCommonMultiple_v(const std::vector<Sett> &setts);

  /**
   * Test for set equivalence.
   *
   * \return true, iff the positions at which this Sett is '1' (on) are the
   *         same as those for which rhs is '1'. Note that having the same
   *         Stripes implies equivalence, but equivalence does not imply the
   *         same Stripes. For example, the Setts with nested Stripes
   *         ((3,0,0)(1,1,0)) and ((2,1,2)) are equivalent, but have different
   *         Stripes.
   * */
  bool equivalent(const Sett &rhs) const;

  /**
   * Throw a descriptive error if rhs is not equivalent to this Sett.
   * */
  void confirmEquivalent(const Sett &rhs) const;

  /**
   * \param rhs Disjoint Setts.
   *
   * \return true iff the positions at which this Sett is '1' (on) are the
   *        same as the union over rhs.
   */
  bool equivalent(const DisjointSetts &rhs) const;

  /**
   * \param rhs A set of disjoint Setts.
   *
   * Throw a descriptive error if the union of rhs is not equivalent to this
   * Sett, otherwise do nothing.
   *
   * */
  void confirmEquivalent(const DisjointSetts &rhs) const;

  /**
   * \return true iff this Sett is always off (always '0'). An example of
   *         when this is when one of the nested Stripes is always off,
   *         although it is also possible to be true when none of the nested
   *         Stripes are always off, but are misaligned. An example:
   *         ((1,2,0),(1,1,1)) is always off.
   * */
  bool alwaysOff() const;

  /**
   * If a Sett has no Stripes, it is always on. But an uncanonicalized Sett
   * can be always on, and have Stripes. An example, ((5,0,3),(4,0,3),(5,5,0))
   * is always on.
   * */
  bool hasStripes() const { return !stripes.empty(); }

  /**
   * Test that this is a superset of rhs.
   *
   * \return true iff (rhs is on at i implies that this on at i).
   * */
  bool contains(const Sett &rhs) const;

  /**
   * \return true iff the intersection with rhs is empty.
   * */
  bool disjoint(const Sett &rhs) const;

  /**
   * \param setts The set to test for pairwise disjointedness.
   *
   * \return false iff the intersection of any 2 elements of setts is
   *         non-empty.
   * */
  static bool disjoint(const std::vector<Sett> &setts);

  /**
   * \param begin The starting index.
   *
   * \return The first index with a '1' starting from index begin.
   *
   * Examples. If this is:
   * ..11..11.1.........1
   * 01234567890123456789
   *
   * find(0)  = 2
   * find(2)  = 2
   * find(5)  = 6
   * find(9)  = 9
   * find(10) = 19
   *
   * Negative values are also allowed, if this is:
   * ..111..1...111..1111.1.11.111.11.1
   *         9876543210123456789
   *
   * find(-8) = -6
   * find(-6) = -6
   * find(-3) = -1
   * find(-1) = -1
   * find(0) = 0
   * find(3) = 3
   * find(4) = 4
   *
   * */
  int64_t find(int64_t begin) const;

  /**
   * Throw a descriptive error if the Setts in rhs are not disjoint.
   * */
  static void confirmDisjoint(const std::vector<Sett> &rhs);

  /**
   * \param setts A set of disjoint Setts
   *
   * \return true if and only if, for all i where this is on, there is at
   * least 1 Sett in setts which is in on at i
   * */
  bool containedIn(const DisjointSetts &setts) const;

  /**
   * Partition this Sett into a set of Setts which repeat with
   * periodicity 's'. Specifically, factorize this Sett into: {p00, p01}
   * ... {pN0, pN1}. such that this Sett is the union of Sett partition,
   * {{p00.getScaled(s), p01}  ... {pN0.getScaled(s), pN1}}.
   * */
  std::vector<std::array<Sett, 2>> unflatten(int64_t s) const;

  /**
   * \param unfs A vector of (prefix, suffix) pairs of Setts
   *
   * \param scale The period to scale the prefix Setts by
   *
   * Example
   * prefix = ((2,3,4),(1,1,0)), suffix ((5,6,7)), s = 10.
   * The "scaled concatenation" is the depth-3 Sett,
   * ((20,30,40), (10,10,0), (5,6,7)).
   *
   * \return The vector, obtained by performing the above scaled concatenation
   * on all (prefix, suffix) pairs, followed by a canonicalization (removal of
   * empty Setts, etc.).
   *
   * */
  static std::vector<Sett>
  scaledConcat(const std::vector<std::array<Sett, 2>> &unfs, int64_t scale);

  /**
   * Counting in the ordered set. Starting from index 0, at what index is the
   * nthOn '1'?
   *
   * \param nthOn The number of '1's to visit before returning.
   *
   * \return The index at which the nthOn '1' appears.
   *
   * Examples:
   *
   * 1...1111
   * getOn(0) = 0
   * getOn(1) = 4.
   *
   * ..11..11...11.1
   * getOn(0) = 2
   * getOn(2) = 6.
   *
   * .11.11.11.11.11.11.11.11.  (2,1,1)
   * 9876543210123456789
   * ========
   * negative
   * getOn(-3) = -4
   * getOn(-2) = -2
   * getOn(-1) = -1
   * getOn( 0) =  1
   * getOn( 1) =  2
   * getOn( 2) =  3.
   *
   * */

  int64_t getOn(int64_t nthOn) const;

  /**
   * \param start The (included) start of the range
   *
   * \param end  The (excluded) end of the range
   *
   * Requirement: start <= end.
   *
   * \return Positions of all '1' values in [start, end), in ascending order
   *
   * */
  std::vector<int64_t> getOns(int64_t start, int64_t end) const;

  /**
   * \return A boolean vector of length \p end - \p start. The value at index
   *         i is `true' if this Sett is on at position start + i, otherwise
   *         it is `false'.
   * */
  std::vector<bool> getBoolOns(int64_t start, int64_t end) const;

  /**
   * Number of '1's (ons) in a contiguous range.
   * */
  int64_t n(int64_t start, int64_t end) const;
  int64_t n(int64_t end) const { return n(0, end); }

  /**
   * Set intersection
   *
   * \param a The first Sett in the intersection
   *
   * \param b The second Sett in the intersection
   *
   * \return The intersection of 2 Setts, represented as a union of
   *         disjoint Setts.
   *
   * Example:
   *  a : 1.1..1.11..1..11.1..1..11......11111.......11.. a
   *  b : .11.11..11..1.11.11.11.1....111111......11..... b
   *  =>  ..1..1..1.....11.1..1..1.......111............. the intersection
   *  */
  static DisjointSetts intersect(const Sett &a, const Sett &b);

  /**
   * Intersect this Sett with rhs
   * */
  DisjointSetts intersect(const Sett &rhs) const;

  /**
   * Set "composition by division"
   *
   * \param sett The Sett to sample from
   *
   * \param filter The positions to sample at
   *
   * \return The sub-sampled Sett.
   *
   * Example:
   *  sett  :  1..1..1.11....1.11...11..11...1.111.11.............11..11
   *  filter:  .11...1.11.11..1.1111.1.1..11.11.1...........1.....111111
   *            ..   1 11 ..  . 1... 1 .  .. 1. 1           .     11..11
   * return  :
   *  =>       ..111...1...1...1.1.11..11
   * */
  static DisjointSetts sample(const Sett &sett, const Sett &filter);
  DisjointSetts sampleAt(const Sett &filter) const;

  /**
   * Set "composition by multiplication"
   *
   * \param scaffold The positions at which to insert
   *
   * \param ink The 0/1s to insert
   *
   * \return The "super-sampled" (with zeros in gaps) Sett.
   *
   * Example:
   *  scaffold : 1..111.1..1.11.1..11.1.11..1.11.1.11.1..1111
   *  ink      : .1..1.11...1.11.1..1.1..1
   *             .  1.. 1  . 11 .  .. 1 .1  1 .1 . .1 .  1..1
   *   =>        ...1...1....11.......1..1..1..1....1....1..1 (super-sampled)
   *
   *  Note, sample(fill(scaffold, ink), scaffold) = ink.
   *  Similarly, fill(indices, sample(x, indices)) is contained in x:
   *
   *  ..111...1111..11...111..1111.....   x
   *  1..11.111...11..11...1.11...11....  indices
   *  .  11 ..1   ..  ..   1 .1   11      sample(x, indices)
   *  ...11...1............1..1...11....  fill(indices, sample(x, indices))
   *
   * In suppary, where f = "filter" and s = "sample",
   *
   *  f(i, s(x,i)) != x,  but
   *  s(f(s, i), s) = i.
   *
   *  */
  static DisjointSetts fill(const Sett &scaffold, const Sett &ink);
  DisjointSetts fillWith(const Sett &ink) const;

  uint64_t recursiveDepth_u64() const { return stripes.size(); }
  int recursiveDepth() const { return static_cast<int>(stripes.size()); }

  /**
   * \param d the depth at which to start copying Stripes
   *
   * \return A Sett constructed from Stripes at depths [d,
   *         recursiveDepth()).
   *
   * Example:
   * If this is ((10,10,2), (4,2,2), (1,2,1)), then
   * fromDepth(1) returns ((4,2,2), (1,2,1)).
   * */
  Sett fromDepth(int d) const;

  const Stripe &atDepth(uint64_t i) const { return stripes[i]; }
  const std::vector<Stripe> &getStripes() const { return stripes; }

  /**
   * Append logging information to ost.
   * */
  void append(std::ostream &ost) const;

  /**
   * \return A Sett identical to this, but with the outermost Stripe (if
   *         this Sett has any Stripes) phase shifted forward by deltaPhase0.
   * */
  Sett phaseShifted(int64_t deltaPhase0) const;

  /**
   * \param s0 The Stripe to prepend
   *
   * \return A Sett identitical to this but with a prepended (i.e.
   *         outermost) Stripe, and an adjusted phase of the first Stripe.
   *
   * Example:
   * If this is ((1,1,1)) and s0 = (4, 2, 1) return ((4, 2, 1),(1, 1, 0)).
   * .1.1.1.1.1.1. this
   * .1111..       s0
   *
   * The adjustment of the first Stripe's phase is a common transformation
   * when nesting Stripes.
   * */
  Sett adjustedPrepend(const Stripe &s0) const;

  /**
   * \param pivot The index at which to rotate this Sett
   *
   * Example
   * If this Sett is ((3, 1, -1)), then reverse(8) is ((3, 1, -2)):
   *             0       8
   *           .111.111.111.111.111
   *         <-----------
   *
   * In this example, reverse(0) is also ((3, 1, -1)).
   *
   *  */
  Sett getReverse(int64_t pivot) const;

  /**
   * \return The union of Setts representing the complement of this
   * (0<->1).
   *
   * Example:
   * 1.11..1.11.11..11.11.1....11.111.1.1 (this)
   * .1..11.1..1..11..1..1.1111..1...1.1. (to return)
   *
   * */
  DisjointSetts getComplement() const;

  /**
   * \param rhs The Sett to subtract from this Sett.
   *
   * \return The intersection of this and the complement of rhs.
   *
   * Example:
   * 11111...11111... this
   * ..11111...11111. rhs
   * 11.....111.....1 rhs's complement
   * 11......11...... returned.
   *
   * The retured Sett is the set of all indices which are '1' in this and
   * not '1' in rhs.
   * */
  DisjointSetts subtract(const Sett &rhs) const;

  /**
   * \return A vector of size smallest-common-multiple of rhs, where element i
   *         is the number of Setts "x" in rhs for which "x" is on at i.
   *  */
  static std::vector<int> getRepeatingOnCount(const std::vector<Sett> &rhs);

  /**
   * Attempt to merge two disjoint Setts.
   * */
  static OptionalSett1 merge(const Sett &, const Sett &);

  /**
   * Attempt to transform two disjoint Setts into two simpler Setts.
   * */
  static OptionalSett2 transfer(const Sett &, const Sett &);

  /**
   * The first depth at which this and rhs have a different Stripe.
   * */
  int depthWhereFirstDifference(const Sett &rhs) const;

  /**
   * \param setts Setts to be canonicalized
   *
   * \return Merged and shortened, but equivalent version of setts.
   * */
  static DisjointSetts canonicalized(const DisjointSetts &setts);

  int64_t period() const { return hasStripes() ? atDepth(0).period() : 1; }

private:
  /**
   * \param p the non-crossing period, it must be a factor of the outermost
   *          Stripe's period.
   *
   * \return A partition of this Sett into 1 or 3 Setts, based on the
   *         first (outermost) Stripe's phase and period. The partition
   *         consists of "overflows" on the edges, if there are any, and the
   *         main repeated section.
   *
   * Example: If this is ((3,2,4)) and p is 20
   * 11..111..111..111..111..111..111..111..111
   * 0                   0                   0
   * ==  -------------  +==  -------------  +==
   * == : incomplete 1s at begin
   * -- : complete 1s in interior
   *  + : incomplete 1s at end
   *
   * Example 2: If this is ((3,2,2)) and p is 20
   * ..111..111..111..111..111..111..111..111
   * 0                   0                   0
   *   -----------------   ------------------
   * in this example, there is no incomplete start or end, so {*this} is
   * returned.
   *
   * */
  DisjointSetts getNonCrossingB(int64_t p) const;

  DisjointSetts getNonCrossingPeriodic(int64_t p, int64_t upper) const;

  std::array<Sett, 2> getPeriodSplit() const;

  // If Sett "b" looks like a slice of the first or final Stripe of "a" at
  // any depth, or first Stripe at any depth, paste them together. Example
  // 11.11.......11.11..., and
  // ......11..........11.
  //
  // + (d, d+1) -> d+1
  //
  static OptionalSett1 mergeA(const Sett &, const Sett &);

  //  Concatentate two Setts at any depth if they fit "seemlessly"
  //  Example:
  //  11111111......11111111..
  //  ........11............11
  //
  // + (d, d) -> d
  //
  static OptionalSett1 mergeB(const Sett &, const Sett &);

  // insert an intermediate Stripe to merge 2 Setts
  // Example:
  // .11......11..... (...)(2, 6, 1)(...)
  // ....11......11.. (...)(2, 6, 4)(...)
  // becomes
  // (...)(5, 3, 1)(2, 1, 0)(...)
  //
  // + (d, d) -> d+1
  //
  static OptionalSett1 mergeC(const Sett &, const Sett &);

  // Example 1:
  // ....11..........11....... (...)(2,10,4)
  // ......1.1.........1.1.... (...)(3,9,6)(1,1,0)
  // (...)(3,9,4) and (...)(1,11,8)
  //
  // Example 2:
  // ......1111........1111........1111.......
  //           11.11       11.11       11.11
  //
  //
  // + (d, d+1) -> (d, d)
  //
  static OptionalSett2 transferA(const Sett &, const Sett &);

  void canonicalize();

  // The recursive implementation of Sett::n
  int64_t nRecursive(uint64_t depth, int64_t start, int64_t end) const;

  // The recursive implementation of Sett::getOns
  std::vector<int64_t>
  getOnsRecurse(uint64_t depth, int64_t start, int64_t end) const;

  /**
   * The recursive implementation of Sett::unflatten.
   *
   * \param period The value which all prefixes must have as a factor of
   *               their periods.
   *
   * \param heightLimit There is only a requirement that the returned vector
   *                    is correct in the range [0, heightLimit) of the
   *                    prefixes. In other words, if this Sett changes outside
   *                    of the range [0, heightLimit*period), the returned
   *                    vector from this function does not need to change.
   *
   * \param depth The recursive depth of the call. Used only for debugging.
   * */
  std::vector<std::array<Sett, 2>>
  unflattenRecurse(int64_t period, int64_t heightLimit, int depth) const;

  // The recursive implementation of Sett::getOn
  int64_t getOnRecurse(int64_t nthOn) const;

  /**
   * The recursive implementation of Sett::intersect, with 3 additional
   * arguments to accelerate the algorithm.
   *
   * \param lowCorrect The intersection is only required to be correct in the
   *                    range [lowCorrect, uppCorrect). See param uppCorrect.
   *
   * \param uppCorrect The intersection is only required to be correct in the
   *                    range [lowCorrect, uppCorrect). See param lowCorrect.
   *
   * \param depth carry the depth of the recursion so that it is known at any
   *              time - only used for debugging.
   *
   *
   * By carrying lowCorrect and uppCorrect down through the recursion, and
   * progressively making the required boud of correctness narrower, the
   * algorithm avoids computing intersections of Setts which will eventually
   * be unused.
   * */
  static DisjointSetts intersectRecurse(const Sett &lhs,
                                        const Sett &rhs,
                                        uint64_t depth,
                                        int64_t lowCorrect,
                                        int64_t uppCorrect);

  // The recursive implementation of Sett::fill
  static DisjointSetts fillRecurse(const Sett &scaffold,
                                   const Sett &ink,
                                   int depth,
                                   int64_t scaffoldUpp);

  /**
   * The recursive implementation of Sett::sample, with 2 additional
   * parameters:
   *
   * \param lowCorrect The sampling is only required to be correct in the
   *                   final range [lowCorrect, uppCorrect).
   *
   * */
  static DisjointSetts sampleRecurse(const Sett &x,
                                     const Sett &filter,
                                     int depth,
                                     int64_t lowCorrect,
                                     int64_t uppCorrect);

  void appendStripes(const std::vector<Stripe> &s) {
    stripes.insert(stripes.end(), s.cbegin(), s.cend());
  }

  void changeFirstStripe(const Stripe &s0) { stripes[0] = s0; }

  void prependStripes(const std::vector<Stripe> &s) {
    stripes.insert(stripes.begin(), s.cbegin(), s.cend());
  }

  DisjointSetts getNonCrossingA(int64_t l0, int64_t u0) const;

  void shiftPhase(int64_t deltaPhase0);

private:
  // The nested Stripes which completely define this Sett
  std::vector<Stripe> stripes;
};

class DisjointSetts {

private:
  // This DisjointSetts is represented by the union of these disjoint Setts
  std::vector<Sett> setts_;

public:
  DisjointSetts() = default;

  /**
   * \param s a vector of disjoint Setts. If the Setts in s are not all
   * disjoint, the behavior is if the object constructed is undefined.
   * */
  explicit DisjointSetts(const std::vector<Sett> &s) : setts_(s) {}

  explicit DisjointSetts(std::vector<Sett> &&s) : setts_(std::move(s)) {}

  DisjointSetts(const Sett &s) : setts_({s}) {}

  decltype(setts_.begin()) begin() { return setts_.begin(); }

  decltype(setts_.cbegin()) cbegin() const { return setts_.cbegin(); }

  decltype(setts_.end()) end() { return setts_.end(); }

  decltype(setts_.cend()) cend() const { return setts_.cend(); }

  size_t size() const { return setts_.size(); }

  bool empty() const { return setts_.empty(); }

  const std::vector<Sett> &get() const { return setts_; }

  std::vector<Sett> &get() { return setts_; }

  const Sett &operator[](size_t i) const { return setts_[i]; }

  Sett &operator[](size_t i) { return setts_[i]; }

  bool equivalent(const DisjointSetts &rhs) const;

  int64_t totalOns(int64_t end) const;
};

std::ostream &operator<<(std::ostream &stream, const Sett &);
std::ostream &operator<<(std::ostream &stream, const std::vector<Sett> &);
std::ostream &operator<<(std::ostream &stream, const DisjointSetts &);
std::ostream &operator<<(std::ostream &stream, const OptionalSett1 &);

} // namespace nest
} // namespace memory
} // namespace poprithms

#endif
