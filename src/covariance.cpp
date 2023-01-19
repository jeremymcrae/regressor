
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

#if defined(__x86_64__)
  #include <immintrin.h>
#endif

#include "covariance.h"

namespace regressor {

// sum array of floats with vectorised operations
//
// We need to load the data into vector registers, but we need to convert the
// float vectors to doubles for numerical stability, otherwise the sum can be
// very inaccurate across many large values. Note that this is not seen if
// using a naive sum of each float in turn, presumably as the CPU uses 64 or 80-bit
// operations.
//
// I tried using __m256 float vectors, and got inaccurate sums with large values.
// I tried Kahan summation (see https://en.wikipedia.org/wiki/Kahan_summation),
// and got same sums using this function, a naive sum, or Kahan summation.
//
/// @param x array of floats
/// @param size size of array
/// @returns sum of array
double fast_sum(float * x, const std::uint32_t & size) {
  size_t i = 0;
  double total = 0.0;

#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx2")) {
    double arr[4];
    __m256d _vals1, _vals2, _vals3, _vals4;
    __m256d _sum1 = {0.0, 0.0, 0.0, 0.0};
    __m256d _sum2 = {0.0, 0.0, 0.0, 0.0};
    __m256d _sum3 = {0.0, 0.0, 0.0, 0.0};
    __m256d _sum4 = {0.0, 0.0, 0.0, 0.0};
    // unroll loops for ~2.7-fold faster summation than with step size of 4
    for (; i + 16 < size; i += 16) {
      // load float vectors and convert to doubles
      _vals1 = _mm256_cvtps_pd(_mm_loadu_ps(&x[i]));
      _vals2 = _mm256_cvtps_pd(_mm_loadu_ps(&x[i + 4]));
      _vals3 = _mm256_cvtps_pd(_mm_loadu_ps(&x[i + 8]));
      _vals4 = _mm256_cvtps_pd(_mm_loadu_ps(&x[i + 12]));

      _sum1 = _mm256_add_pd(_sum1, _vals1);
      _sum2 = _mm256_add_pd(_sum2, _vals2);
      _sum3 = _mm256_add_pd(_sum3, _vals3);
      _sum4 = _mm256_add_pd(_sum4, _vals4);
    }
    _mm256_storeu_pd(arr, _sum1);
    total += arr[0] + arr[1] + arr[2] + arr[3];
    _mm256_storeu_pd(arr, _sum2);
    total += arr[0] + arr[1] + arr[2] + arr[3];
    _mm256_storeu_pd(arr, _sum3);
    total += arr[0] + arr[1] + arr[2] + arr[3];
    _mm256_storeu_pd(arr, _sum4);
    total += arr[0] + arr[1] + arr[2] + arr[3];
  }
#endif

  // include the reimainder not used during vectorised sum
  for ( ; i < size; i++) {
    total += x[i];
  }
  return total;
}

/// get means of two samed-sized float arrays
///
/// @param x array of x-values
/// @param size_x size of x-values array
/// @param y array of y-values
/// @param size_y size of y-values array
/// @returns covmeans struct with x and y means
covmeans covariance_means(float * x, const std::uint32_t & size_x, float * y, const std::uint32_t & size_y) {
  if (size_x != size_y) {
    throw std::invalid_argument("arrays do not have same length");
  }

  return {fast_sum(x, size_x) / size_x, fast_sum(y, size_y) / size_x};
}

/// get covariance values from two same-sized float arrays.
///
/// Use arrays of x-values and y-values to get dot products of (x - xmean)^2,
/// (y - ymean)^2, and (x - xmean) * (y - ymean). Alos returns means, to avoid
/// duplicating mean calculations in the linregress function calling this.
/// This uses AVX operations to quicly get each dot product.
///
/// @param x array of x-values
/// @param size_x size of x-values array
/// @param y array of y-values
/// @param size_y size of y-values array
/// @returns covmeans struct with sums of squares for x^2, x*y, y^2 and averages
covs covariance(float * x, const std::uint32_t & size_x, float * y, const std::uint32_t & size_y) {
  if (size_x != size_y) {
    throw std::invalid_argument("arrays do not have same length");
  }
  if (size_x < 2000) {
    // if the arrays are small enough, use a function which converts the floats
    // to doubles during calculations, in order to avoid an edge case of discrepant
    // results with small arrays when the arrays are near identical, and differ
    // by some scalar increment.
    return covariance_higher_precision(x, size_x, y, size_y);
  }

  covmeans mu = covariance_means(x, size_x, y, size_y);
  
  double s_xx = 0.0;
  double s_xy = 0.0;
  double s_yy = 0.0;
  size_t i = 0;
  double ddof = (double) size_x;

#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx2")) {
    __m256 x_means = {(float) mu.x, (float) mu.x, (float) mu.x, (float) mu.x, (float) mu.x, (float) mu.x, (float) mu.x, (float) mu.x};
    __m256 y_means = {(float) mu.y, (float) mu.y, (float) mu.y, (float) mu.y, (float) mu.y, (float) mu.y, (float) mu.y, (float) mu.y};
    
    __m256 x_vals, y_vals;
    __m256 _s_xx = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    __m256 _s_xy = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    __m256 _s_yy = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    // calculate sums of x^2, xy, and y**2 in batches of 16
    for ( ; i + 16 < size_x; i += 8) {
      x_vals = _mm256_loadu_ps(&x[i]);
      y_vals = _mm256_loadu_ps(&y[i]);

      x_vals = _mm256_sub_ps(x_vals, x_means);
      y_vals = _mm256_sub_ps(y_vals, y_means);

      _s_xx = _mm256_fmadd_ps(x_vals, x_vals, _s_xx);
      _s_xy = _mm256_fmadd_ps(x_vals, y_vals, _s_xy);
      _s_yy = _mm256_fmadd_ps(y_vals, y_vals, _s_yy);
    }

    // add in vectorized sums
    float arr[8];
    _mm256_storeu_ps(arr, _s_xx);
    s_xx += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
    _mm256_storeu_ps(arr, _s_xy);
    s_xy += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
    _mm256_storeu_ps(arr, _s_yy);
    s_yy += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
  }
#endif

  // include samples not in vectorization
  for ( ; i < size_x; i++) {
    s_xx += ((double) x[i] - (double) mu.x) * ((double) x[i] - (double) mu.x);
    s_xy += ((double) x[i] - (double) mu.x) * ((double) y[i] - (double) mu.y);
    s_yy += ((double) y[i] - (double) mu.y) * ((double) y[i] - (double) mu.y);
  }

  return {mu, s_xx / ddof, s_xy / ddof, s_xy / ddof, s_yy / ddof, size_x};
}

/// higher precision covariance used with small arrays.
///
/// This only differs from covariance() when arrays are a) small, b) perfectly
/// correlated  and c) differ by some scalar. This works as per covariance(), but
/// casts the floats to doubles for extra precision during calculations.
///
/// @param x array of x-values
/// @param size_x size of x-values array
/// @param y array of y-values
/// @param size_y size of y-values array
/// @returns covmeans struct with sums of squares for x^2, x*y, y^2 and averages
covs covariance_higher_precision(float *x, const std::uint32_t &size_x, float *y, const std::uint32_t &size_y)
{
  if (size_x != size_y) {
    throw std::invalid_argument("arrays do not have same length");
  }
  covmeans mu = covariance_means(x, size_x, y, size_y);

  size_t i = 0;
  double s_xx = 0.0;
  double s_xy = 0.0;
  double s_yy = 0.0;
  double ddof = (double) size_x;

#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx2")) {
    __m256d x_means = {mu.x, mu.x, mu.x, mu.x};
    __m256d y_means = {mu.y, mu.y, mu.y, mu.y};

    __m128 x128, y128;
    __m256d x_vals, y_vals;
    __m256d _s_xx = {0.0, 0.0, 0.0, 0.0};
    __m256d _s_xy = {0.0, 0.0, 0.0, 0.0};
    __m256d _s_yy = {0.0, 0.0, 0.0, 0.0};

    // calculate sums of x^2, xy, and y**2 in batches of 8
    for (; i + 4 < size_x; i += 4) {
      x128 = _mm_loadu_ps(&x[i]);
      y128 = _mm_loadu_ps(&y[i]);

      x_vals = _mm256_cvtps_pd(x128);
      y_vals = _mm256_cvtps_pd(y128);

      x_vals = _mm256_sub_pd(x_vals, x_means);
      y_vals = _mm256_sub_pd(y_vals, y_means);

      _s_xx = _mm256_fmadd_pd(x_vals, x_vals, _s_xx);
      _s_xy = _mm256_fmadd_pd(x_vals, y_vals, _s_xy);
      _s_yy = _mm256_fmadd_pd(y_vals, y_vals, _s_yy);
    }

    // add in vectorized sums
    double arr[4];
    _mm256_storeu_pd(arr, _s_xx);
    s_xx += arr[0] + arr[1] + arr[2] + arr[3] ;
    _mm256_storeu_pd(arr, _s_xy);
    s_xy += arr[0] + arr[1] + arr[2] + arr[3] ;
    _mm256_storeu_pd(arr, _s_yy);
    s_yy += arr[0] + arr[1] + arr[2] + arr[3] ;
  }
#endif

  // include samples not in vectorization
  for (; i < size_x; i++) {
    s_xx += ((double) x[i] - (double) mu.x) * ((double) x[i] - (double) mu.x);
    s_xy += ((double) x[i] - (double) mu.x) * ((double) y[i] - (double) mu.y);
    s_yy += ((double) y[i] - (double) mu.y) * ((double) y[i] - (double) mu.y);
  }

  return {mu, s_xx / ddof, s_xy / ddof, s_xy / ddof, s_yy / ddof, size_x};
}

} // regressor namespace
