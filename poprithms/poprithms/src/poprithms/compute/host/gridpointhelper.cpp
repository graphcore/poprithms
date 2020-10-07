// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "./include/gridpointhelper.hpp"

#include <algorithm>
#include <numeric>

#include <poprithms/compute/host/error.hpp>

namespace poprithms {
namespace compute {
namespace host {

using Row     = GridPointHelper::Row;
using Rows    = std::vector<Row>;
using Column  = GridPointHelper::Column;
using Columns = std::vector<Column>;

namespace {
auto getMaxRow(const Rows &rows) {
  return std::accumulate(rows.cbegin(),
                         rows.cend(),
                         Row(0),
                         [](auto a, auto b) { return std::max(a, b); });
}

auto getMaxColumn(const Columns &columns) {
  return std::accumulate(columns.cbegin(),
                         columns.cend(),
                         Column(0),
                         [](auto a, auto b) { return std::max(a, b); });
}
} // namespace

bool GridPointHelper::allUnique(const Rows &rows, const Columns &columns) {

  const auto nPoints = columns.size();
  if (rows.size() != nPoints) {
    throw error("rows and columns must be same size in allUnique.");
  }

  // Get grid dimensions.
  const auto maxRow    = getMaxRow(rows);
  const auto maxColumn = getMaxColumn(columns);

  // [row][column] map of already seen points
  std::vector<std::vector<bool>> alreadySeen(maxRow + 1);
  for (uint64_t i = 0; i < nPoints; ++i) {
    const auto row        = rows[i];
    const auto column_u64 = static_cast<uint64_t>(columns[i]);
    if (alreadySeen[row].empty()) {
      alreadySeen[row].resize(maxColumn + 1, false);
      alreadySeen[row][column_u64] = true;
    } else if (!alreadySeen[row][column_u64]) {
      alreadySeen[row][column_u64] = true;
    } else {
      return false;
    }
  }
  return true;
}

std::vector<std::tuple<Row, Column>>
GridPointHelper::getUnique(const Rows &rows, const Columns &columns) {

  const auto nPoints = columns.size();
  if (rows.size() != nPoints) {
    throw error("rows and columns must be same size in getUnique");
  }

  std::vector<std::tuple<Row, Column>> unq;

  const auto maxRow    = getMaxRow(rows);
  const auto maxColumn = getMaxColumn(columns);
  std::vector<std::vector<bool>> alreadySeen(maxRow + 1);
  for (uint64_t i = 0; i < nPoints; ++i) {
    const auto row        = rows[i];
    const auto column_u64 = static_cast<uint64_t>(columns[i]);
    if (alreadySeen[row].empty()) {
      alreadySeen[row].resize(maxColumn + 1, false);
      alreadySeen[row][column_u64] = true;
      unq.push_back({row, column_u64});
    } else if (!alreadySeen[row][column_u64]) {
      alreadySeen[row][column_u64] = true;
      unq.push_back({row, column_u64});
    }
  }
  return unq;
}

} // namespace host
} // namespace compute
} // namespace poprithms
