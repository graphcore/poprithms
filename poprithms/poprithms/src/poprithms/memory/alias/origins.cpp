// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <ostream>
#include <vector>

#include <memory/alias/error.hpp>

#include <poprithms/memory/alias/origins.hpp>

namespace poprithms {
namespace memory {
namespace alias {

static_assert(std::is_nothrow_move_constructible<Origins>::value,
              "Expect Origins to be nothrow move constructible");

bool Origins::isAliasedTo(const Origins &rhs) const {

  std::vector<AllocId> commonAllocIds;
  const auto v1 = getAllocIds();
  const auto v2 = rhs.getAllocIds();
  std::set_intersection(v1.cbegin(),
                        v1.cend(),
                        v2.cbegin(),
                        v2.cend(),
                        std::back_inserter(commonAllocIds));

  for (auto ai : commonAllocIds) {
    const auto &regs0 = at(ai);
    const auto &regs1 = rhs.at(ai);
    for (const auto &dj0 : regs0) {
      for (const auto &dj1 : regs1) {
        if (!dj0.disjoint(dj1)) {
          return true;
        }
      }
    }
  }
  return false;
}

std::unique_ptr<Origins> Origins::clone() const {
  return std::make_unique<Origins>(*this);
}

void Origins::append(std::ostream &os) const {
  for (auto &[k, v] : oMap) {
    os << "[" << k << "]:(";
    for (const auto &regs : v) {
      os << regs;
    }
    os << ")\n";
  }
}

std::ostream &operator<<(std::ostream &os, const Origins &o) {
  o.append(os);
  return os;
}

void Origins::insert(AllocId id, const DisjointRegions &regs) {
  const auto found = oMap.find(id);
  if (found == oMap.cend()) {
    oMap.insert({id, {regs}});
  } else {
    found->second.push_back(regs);
  }
  incrementSumTotalRegionSizes(regs.totalElms());
}

void Origins::insert(const Origins &oris) {
  for (const auto &[k, v] : oris.oMap) {
    for (const auto &v2 : v) {
      insert(k, v2);
    }
  }
}

void Origins::incrementSumTotalRegionSizes(uint64_t n) {
  sumTotalRegionSizes += n;
  if (n > shape.nelms_u64()) {
    std::ostringstream oss;
    oss << "Error in Origins::incrementSumTotalRegionSizes(" << n << ") "
        << "where shape has " << shape.nelms_u64() << " elements. "
        << "There cannot be more allocations than elements, "
        << "only as many or fewer (fewer when self aliased)."
        << " By incrementing sumTotalRegionSizes by " << n
        << ", the total becomes " << sumTotalRegionSizes << ". ";
    throw error(oss.str());
  }
}

std::vector<AllocId> Origins::getAllocIds() const {
  std::vector<AllocId> ids;
  ids.reserve(oMap.size());
  for (const auto &kv : oMap) {
    ids.push_back(kv.first);
  }
  return ids;
}

bool Origins::containsAliases() const {

  // This Origins object is storing the allocations of all elements of a
  // Tensor with shape.nelms(). If the total number of registed allocation
  // addresses is less than shape.nelms_u64(), then there are aliases (we
  // assume that all elements have had origins trace).
  if (sumTotalRegionSizes < shape.nelms_u64()) {
    return true;
  }

  // Total registered allocations is shape.nelms_u64(). Are they all actually
  // distinct?
  for (const auto &[id, allRegs] : oMap) {
    (void)id;
    for (auto iter0 = allRegs.cbegin(); iter0 != allRegs.cend(); ++iter0) {
      for (auto iter1 = std::next(iter0, 1); iter1 != allRegs.cend();
           ++iter1) {
        if (!iter0->disjoint(*iter1)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool Origins::isRowMajorSetContiguous() const {

  const std::vector<DisjointRegions> *drp{nullptr};
  for (const auto &[id, regions] : oMap) {
    (void)id;
    if (std::any_of(regions.cbegin(), regions.cend(), [](const auto &r) {
          return !r.empty();
        })) {

      // more than 1 non-empty allocation, return false.
      if (drp) {
        return false;
      }
      drp = &regions;
    }
  }
  // Empty Tensor, always row major set contiguous.
  if (!drp) {
    return true;
  }

  int64_t globalLow = drp->front().shape().nelms();
  int64_t globalUpp = 0;

  for (const auto &regs : *drp) {
    for (auto reg : regs.get()) {
      const auto sett = reg.flatten().sett(0);
      const auto nOn  = sett.n(0, reg.shape().nelms());
      if (nOn != 0) {
        globalLow = std::min(globalLow, sett.getOn(0));
        globalUpp = std::max(globalUpp, sett.getOn(nOn - 1)) + 1;
        if (globalUpp - globalLow > shape.nelms()) {
          return false;
        }
      }
    }
  }

  if (globalUpp - globalLow < shape.nelms()) {
    return false;
  }

  // If self-aliases, return false.
  if (containsAliases()) {
    return false;
  }

  if (globalUpp - globalLow != shape.nelms()) {
    std::ostringstream oss;
    oss << "Logic error in Origins::isRowMajorSetContiguous. "
        << "globalUpp = " << globalUpp << ", globalLow = " << globalLow
        << ", shape.nelms() = " << shape.nelms()
        << " with no self-aliasing. ";
    throw error(oss.str());
  }

  return true;
}

Origins Origins::remap(const std::vector<uint64_t> &toNew) const {
  std::map<AllocId, std::vector<DisjointRegions>> newMap;

  for (const auto &[k, v] : oMap) {
    const auto n       = toNew.at(k.get());
    newMap[AllocId(n)] = v;
  }
  auto remapped = *this;
  remapped.oMap = newMap;
  return remapped;
}

} // namespace alias
} // namespace memory
} // namespace poprithms
