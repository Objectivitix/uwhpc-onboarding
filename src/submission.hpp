#pragma once

#include <cstddef>
#include <new>
#include <vector>
#include <immintrin.h>

constexpr std::size_t VECTORIZED_COLUMN_STEP = sizeof(__m256d) / sizeof(double);

template <typename T, std::size_t Alignment = 32>
struct AlignedAllocator {
  using value_type = T;

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  T* allocate(std::size_t n) {
    void* ptr = ::operator new[](n * sizeof(T), std::align_val_t{Alignment});
    return static_cast<T*>(ptr);
  }

  void deallocate(T* ptr, std::size_t) noexcept {
    ::operator delete[](ptr, std::align_val_t{Alignment});
  }
};

template <typename T, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<T, Alignment>&) noexcept {
  return true;
}

template <typename T, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>& lhs, const AlignedAllocator<T, Alignment>& rhs) noexcept {
  return !(lhs == rhs);
}

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  std::vector<double, AlignedAllocator<double, 32>> cells_;

public:
  Grid(const std::size_t rows, const std::size_t cols)
    : rows_{rows}
    , cols_{cols}
    , stride_{(cols + VECTORIZED_COLUMN_STEP - 1) / VECTORIZED_COLUMN_STEP * VECTORIZED_COLUMN_STEP}
    , cells_(rows * stride_, 0.0)
  { }

  double& operator()(const std::size_t row, const std::size_t col) {
    return cells_[row * stride_ + col];
  }

  double operator()(const std::size_t row, const std::size_t col) const {
    return cells_[row * stride_ + col];
  }

  const double* data() const { return cells_.data(); }
  double* data() { return cells_.data(); }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  std::size_t stride() const { return stride_; }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows{old_grid.rows()};
  const std::size_t cols{old_grid.cols()};

  const double* __restrict__ old_cells{old_grid.data()};
  double* __restrict__ new_cells{new_grid.data()};
  const std::size_t stride{old_grid.stride()};

  __m256d _FOUR = _mm256_set1_pd(4.0);
  __m256d _EIGHTH = _mm256_set1_pd(0.125);

  #pragma omp parallel
  {
    #pragma omp for
    for (std::size_t row = 1; row < rows - 1; ++row) {
      __m256d _above, _below, _left, _right, _curr;
      __m256d _above_plus_below, _left_plus_right;
      __m256d _all_around, _scaled_result, _final_result;

      const std::size_t row_offset{row * stride};
      const std::size_t above_offset{row_offset - stride};
      const std::size_t below_offset{row_offset + stride};

      for (std::size_t col{1}; col < cols - 1; col += VECTORIZED_COLUMN_STEP) {
        // each line below loads 4 packed 64-bit floating-point
        // values starting from an unaligned memory address
        _above = _mm256_loadu_pd(old_cells + above_offset + col);
        _below = _mm256_loadu_pd(old_cells + below_offset + col);
        _left = _mm256_loadu_pd(old_cells + row_offset + col - 1);
        _right = _mm256_loadu_pd(old_cells + row_offset + col + 1);
        _curr = _mm256_loadu_pd(old_cells + row_offset + col);

        // calculate the weighted sums with as few hardware
        // instructions as possible using fused multiply-add (FMA)
        _above_plus_below = _mm256_add_pd(_above, _below);
        _left_plus_right = _mm256_add_pd(_left, _right);
        _all_around = _mm256_add_pd(_above_plus_below, _left_plus_right);
        _scaled_result = _mm256_fmadd_pd(_curr, _FOUR, _all_around);
        _final_result = _mm256_mul_pd(_scaled_result, _EIGHTH);

        // store the sums (4 packed 64-bit floating-point values)
        _mm256_storeu_pd(new_cells + row_offset + col, _final_result);
      }
    }

    // implicit barrier here: all vectorised work is finished;
    // there will be no race conditions (from the out-of-bounds `storeu`
    // in the inner loop above) when copying the side-boundary values
    // to `new_grid`
    #pragma omp for
    for (std::size_t row = 0; row < rows; ++row) {
      new_grid(row, 0) = old_grid(row, 0);
      new_grid(row, cols - 1) = old_grid(row, cols - 1);
    }
  }

  for (std::size_t col{}; col < cols; ++col) {
    new_grid(0, col) = old_grid(0, col);
    new_grid(rows - 1, col) = old_grid(rows - 1, col);
  }
}
