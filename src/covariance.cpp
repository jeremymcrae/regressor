
#include <cmath>
#include <cstdint>
#include <exception>
#include <algorithm>

#include <immintrin.h>

#include "covariance.h"

namespace regressor {

/// estimate mean by sampling from the values
///
/// this estimates the mean from  a subsample of values. The estimate is
/// returned when a subsequent batch doesn't change the estimated mean by some
/// tolerance.
///
/// This can be 2X faster than calculating the average from all values.
///
/// @param vals array of 32-bit floats
/// @param size size of vals array
/// @param tol tolerance for accepting the estimated mean from batches
/// @returns estimated mean as float
float estimate_mean(float * vals, const std::uint32_t & size, float tol=1e-4) {
  
  // figure out how many samples to take per batch. If the array has less than
  // 1000 entries use the full array length, and use each item. If the array has
  // >1000 items, use at least 1000 samples per batch, or 1% of the array length,
  // whichever is greater. Making it at least 1% of the array length ensures the
  // estimate is close to the true mean in large arrays (>10 million items).
  long n_sampled = std::max((long) size / 100, (long) 1000);
  size_t batchsize = std::min(n_sampled, (long) size);
  size_t stride = std::max(size / batchsize, (unsigned long) 1);
  
  float total = 0;
  float mean = 0.0;
  float current;
  float delta;
  
  // TODO: this could probably also be vectorised to reduce memory accesses
  for (size_t i=0; i<stride; i++) {
    for (size_t j=i; j<size; j += stride) {
      total += vals[j];
    }
    current = total / ((i + 1) * batchsize);
    delta = std::abs(1.0 - current / mean);
    if (mean != 0.0 & delta < tol) {
      return current;
    }
    mean = current;
  }
  
  return mean;
}

/// get means of two samed-sized float arrays
///
/// This estimates the mean from a subsample of values, if allowed. Otherwise
/// the mean is calculated from the full arrays with AVX operations.
/// It's slightly faster to sum both arrays at once, whch I think is due to
/// higher throughput of AVX operations
///
/// @param x array of x-values
/// @param size_x size of x-values array
/// @param y array of y-values
/// @param size_y size of y-values array
/// @param sampled whether to estimate the means of the arrays
/// @returns covmeans struct with x and y means
covmeans covariance_means(float * x, const std::uint32_t & size_x, float * y, const std::uint32_t & size_y, bool & sampled) {
  if (size_x != size_y) {
    throw std::invalid_argument("arrays do not have same length");
  }
  
  if (sampled) {
    // use estimated mean for faster speed
    return {estimate_mean(x, size_x), estimate_mean(y, size_y)};
  }
  
  __m256 x_vals, y_vals;
  __m256 x_sum = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  __m256 y_sum = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  size_t i = 0;
  for ( ; i + 8 < size_x; i += 8) {
    x_vals = _mm256_loadu_ps(&x[i]);
    y_vals = _mm256_loadu_ps(&y[i]);
    x_sum = _mm256_add_ps(x_sum, x_vals);
    y_sum = _mm256_add_ps(y_sum, y_vals);
  }

  // sum both arrays
  float arr[8];
  float x_mu = 0.0;
  float y_mu = 0.0;
  _mm256_storeu_ps(arr, x_sum);
  x_mu += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
  _mm256_storeu_ps(arr, y_sum);
  y_mu += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

  // include the reimainder not used during vectorised sum
  for ( ; i < size_x; i++) {
    x_mu += x[i];
    y_mu += y[i];
  }
  
  return {x_mu / size_x, y_mu / size_x};
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
/// @param sampled whether to estimate the means of the arrays
/// @returns covmeans struct with sums of squares for x^2, x*y, y^2 and averages
covs covariance(float * x, const std::uint32_t & size_x, float * y, const std::uint32_t & size_y, bool sampled) {
  if (size_x != size_y) {
    throw std::invalid_argument("arrays do not have same length");
  }
  covmeans mu = covariance_means(x, size_x, y, size_y, sampled);
  
  __m256 x_means = {mu.x, mu.x, mu.x, mu.x, mu.x, mu.x, mu.x, mu.x};
  __m256 y_means = {mu.y, mu.y, mu.y, mu.y, mu.y, mu.y, mu.y, mu.y};
  
  __m256 x_vals, y_vals;
  __m256 x_vals2, y_vals2;
  __m256 _s_xx = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  __m256 _s_xy = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  __m256 _s_yy = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
  // calculate sums of x^2, xy, and y**2 in batches of 16
  size_t i = 0;
  for ( ; i + 16 < size_x; i += 16) {
    x_vals = _mm256_loadu_ps(&x[i]);
    y_vals = _mm256_loadu_ps(&y[i]);
    
    x_vals = _mm256_sub_ps(x_vals, x_means);
    y_vals = _mm256_sub_ps(y_vals, y_means);
    
    // load a second set of values to improve throughput
    x_vals2 = _mm256_loadu_ps(&x[i + 8]);
    y_vals2 = _mm256_loadu_ps(&y[i + 8]);
    
    _s_xx = _mm256_fmadd_ps(x_vals, x_vals, _s_xx);
    _s_xy = _mm256_fmadd_ps(x_vals, y_vals, _s_xy);
    _s_yy = _mm256_fmadd_ps(y_vals, y_vals, _s_yy);
    
    x_vals2 = _mm256_sub_ps(x_vals2, x_means);
    y_vals2 = _mm256_sub_ps(y_vals2, y_means);
    
    _s_xx = _mm256_fmadd_ps(x_vals2, x_vals2, _s_xx);
    _s_xy = _mm256_fmadd_ps(x_vals2, y_vals2, _s_xy);
    _s_yy = _mm256_fmadd_ps(y_vals2, y_vals2, _s_yy);
  }
  
  // include samples not in vectorization
  double s_xx = 0.0;
  double s_xy = 0.0;
  double s_yy = 0.0;
  for ( ; i < size_x; i++) {
    s_xx += (x[i] - mu.x) * (x[i] - mu.x);
    s_xy += (x[i] - mu.x) * (y[i] - mu.y);
    s_yy += (y[i] - mu.y) * (y[i] - mu.y);
  }
  
  // add in vectorized sums
  double ddof = (double) size_x;
  float arr[8];
  _mm256_storeu_ps(arr, _s_xx);
  s_xx += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
  _mm256_storeu_ps(arr, _s_xy);
  s_xy += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
  _mm256_storeu_ps(arr, _s_yy);
  s_yy += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
  
  return {mu, s_xx / ddof, s_xy / ddof, s_xy / ddof, s_yy / ddof, size_x};
}

} // regressor namespace
