#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  using value_type = std::int64_t;
  static constexpr unsigned fraction_bits_{32};
  static constexpr value_type scale_{value_type{1} << fraction_bits_};

  class CellProxy {
  private:
    value_type& value_;

  public:
    explicit CellProxy(value_type& value) : value_{value} { }

    CellProxy& operator=(const double value) {
      value_ = static_cast<value_type>(std::llround(value * scale_));
      return *this;
    }

    CellProxy& operator=(const CellProxy& other) {
      value_ = other.value_;
      return *this;
    }

    operator double() const {
      return static_cast<double>(value_) / scale_;
    }
  };

  std::size_t rows_;
  std::size_t cols_;
  std::vector<value_type> cells_;

  value_type& encoded(const std::size_t row, const std::size_t col) {
    return cells_[row * cols_ + col];
  }

  const value_type& encoded(const std::size_t row, const std::size_t col) const {
    return cells_[row * cols_ + col];
  }

  friend void apply_stencil(const Grid& old_grid, Grid& new_grid);

public:
  Grid(const std::size_t rows, const std::size_t cols)
    : rows_{rows}
    , cols_{cols}
    , cells_(rows * cols, 0)
  { }

  CellProxy operator()(const std::size_t row, const std::size_t col) {
    return CellProxy{encoded(row, col)};
  }

  double operator()(const std::size_t row, const std::size_t col) const {
    return static_cast<double>(encoded(row, col)) / scale_;
  }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows{old_grid.rows()};
  const std::size_t cols{old_grid.cols()};

  for (std::size_t row{}; row < rows; ++row) {
    new_grid.encoded(row, 0) = old_grid.encoded(row, 0);
    new_grid.encoded(row, cols - 1) = old_grid.encoded(row, cols - 1);
  }

  for (std::size_t col{}; col < cols; ++col) {
    new_grid.encoded(0, col) = old_grid.encoded(0, col);
    new_grid.encoded(rows - 1, col) = old_grid.encoded(rows - 1, col);
  }

  #pragma omp parallel for
  for (std::size_t row = 1; row < rows - 1; ++row) {
    for (std::size_t col{1}; col < cols - 1; ++col) {
      new_grid.encoded(row, col) = (
        (old_grid.encoded(row, col) << 2)
        + old_grid.encoded(row - 1, col)
        + old_grid.encoded(row + 1, col)
        + old_grid.encoded(row, col - 1)
        + old_grid.encoded(row, col + 1)
      ) >> 3;
    }
  }
}
