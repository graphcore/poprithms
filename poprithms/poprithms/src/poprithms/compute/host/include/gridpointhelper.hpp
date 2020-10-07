// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_GRIDPOINTHELPER_HPP
#define POPRITHMS_COMPUTE_HOST_GRIDPOINTHELPER_HPP

#include <tuple>
#include <vector>

namespace poprithms {
namespace compute {
namespace host {
/**
 * A class to help the ViewData class with functionality which is not
 * template-parameter specific.
 */
class GridPointHelper {

public:
  using Row  = uint64_t;
  using Rows = std::vector<Row>;

  using Column  = int64_t;
  using Columns = std::vector<Column>;

  using Coord  = std::tuple<Row, Column>;
  using Coords = std::vector<Coord>;

  /**
   * Determine if all elements in a 2-D grid are unique
   *
   * \param rows  The rows of the elements
   * \param columns The columns of the elements
   *
   * \p rows and \p columns must be the same size
   *
   * \return true iff there are no duplicate (row, column) entries.
   *         Specifically, returns false if there exists i and i', i != i',
   *         s.t. rows[i] == rows[i'] and columns[i] == columns[i'].
   * */
  static bool allUnique(const Rows &rows, const Columns &columns);

  /**
   * Get all unique elements in a 2-D grid
   *
   * \param rows  The rows of the elements
   * \param columns The columns of the elements
   *
   * \p rows and \p columns must be the same size
   *
   * \return (row, column) pairs without duplicates.
   *
   * */
  static Coords getUnique(const Rows &rows, const Columns &columns);
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
