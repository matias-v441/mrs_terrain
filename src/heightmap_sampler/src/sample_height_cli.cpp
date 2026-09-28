// Command-line counterpart of examples/sampler.py, for checking that this
// library reproduces the reference implementation exactly:
//
//     sample_height <lon> <lat> [dataset_dir]
//
// With no dataset directory it samples the dataset installed into this
// package's share directory.  Heights are printed at full double precision so
// the output can be diffed against the Python sampler; a point the dataset
// does not cover prints "nan".

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "heightmap_sampler/height_sampler.hpp"

int main(int argc, char ** argv)
{
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr, "usage: %s <lon> <lat> [dataset_dir]\n", argv[0]);
    return 2;
  }

  try {
    const double longitude = std::stod(argv[1]);
    const double latitude = std::stod(argv[2]);

    heightmap_sampler::HeightSampler sampler = argc == 4 ?
      heightmap_sampler::HeightSampler(std::string(argv[3])) :
      heightmap_sampler::HeightSampler();

    const auto height = sampler.sample(latitude, longitude);
    // A point the dataset does not cover is an answer, not a failure, so this
    // exits 0 and prints "nan". A non-zero exit means the dataset or the
    // arguments were unusable.
    if (!height.has_value()) {
      std::printf("nan\n");
      return 0;
    }
    std::printf("%.17g\n", *height);
    return 0;
  } catch (const std::exception & error) {
    std::fprintf(stderr, "sample_height: %s\n", error.what());
    return 2;
  }
}
