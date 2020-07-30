#ifndef REGRESSOR_COVARIANCE_H_
#define REGRESSOR_COVARIANCE_H_

#include <cstdint>

namespace regressor {

struct covmeans {
  float x;
  float y;
};

struct covs {
  covmeans avg;
  double s_xx;
  double s_xy;
  double s_yx;
  double s_yy;
  std::uint32_t size;
};

// get covariance values from two same-sized float arrays
covs covariance(float * x, const std::uint32_t & size_x, float * y, const std::uint32_t & size_y, bool sampled);

} // namespace regressor

#endif  // REGRESSOR_COVARIANCE_H_
