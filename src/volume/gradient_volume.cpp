#include "gradient_volume.h"
#include <algorithm>
#include <exception>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <gsl/span>

namespace volume {

// Compute the maximum magnitude from all gradient voxels
static float computeMaxMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::max_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute the minimum magnitude from all gradient voxels
static float computeMinMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::min_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute a gradient volume from a volume
static std::vector<GradientVoxel> computeGradientVolume(const Volume& volume)
{
    const auto dim = volume.dims();

    std::vector<GradientVoxel> out(static_cast<size_t>(dim.x * dim.y * dim.z));
    for (int z = 1; z < dim.z - 1; z++) {
        for (int y = 1; y < dim.y - 1; y++) {
            for (int x = 1; x < dim.x - 1; x++) {
                const float gx = (volume.getVoxel(x + 1, y, z) - volume.getVoxel(x - 1, y, z)) / 2.0f;
                const float gy = (volume.getVoxel(x, y + 1, z) - volume.getVoxel(x, y - 1, z)) / 2.0f;
                const float gz = (volume.getVoxel(x, y, z + 1) - volume.getVoxel(x, y, z - 1)) / 2.0f;

                const glm::vec3 v { gx, gy, gz };
                const size_t index = static_cast<size_t>(x + dim.x * (y + dim.y * z));
                out[index] = GradientVoxel { v, glm::length(v) };
            }
        }
    }
    return out;
}

GradientVolume::GradientVolume(const Volume& volume)
    : m_dim(volume.dims())
    , m_data(computeGradientVolume(volume))
    , m_minMagnitude(computeMinMagnitude(m_data))
    , m_maxMagnitude(computeMaxMagnitude(m_data))
{
}

float GradientVolume::maxMagnitude() const
{
    return m_maxMagnitude;
}

float GradientVolume::minMagnitude() const
{
    return m_minMagnitude;
}

glm::ivec3 GradientVolume::dims() const
{
    return m_dim;
}

// This function returns a gradientVoxel at coord based on the current interpolation mode.
GradientVoxel GradientVolume::getGradientInterpolate(const glm::vec3& coord) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getGradientNearestNeighbor(coord);
    }
    case InterpolationMode::Linear: {
        return getGradientLinearInterpolate(coord);
    }
    case InterpolationMode::Cubic: {
        // No cubic in this case, linear is good enough for the gradient.
        return getGradientLinearInterpolate(coord);
    }
    default: {
        throw std::exception();
    }
    };
}

// This function returns the nearest neighbour given a position in the volume given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
GradientVoxel GradientVolume::getGradientNearestNeighbor(const glm::vec3& coord) const
{
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(m_dim))))
        return { glm::vec3(0.0f), 0.0f };

    auto roundToPositiveInt = [](float f) {
        return static_cast<int>(f + 0.5f);
    };

    return getGradient(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// ======= TODO : IMPLEMENT ========
// Returns the trilinearly interpolated gradinet at the given coordinate.
// Use the linearInterpolate function that you implemented below.
GradientVoxel GradientVolume::getGradientLinearInterpolate(const glm::vec3& coord) const
{
    // check whether the point is inside the frame, and if not return 0.
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThan(coord, glm::vec3(m_dim.x - 1, m_dim.y - 1, m_dim.z - 1)))) {
        return { glm::vec3(0.0f), 0.0f };
    }

    /*
       Gradient interpolation example (Variable names based on this art)
           xfac
        a00--R00-a01
        |    |   | \
        | ---R0--|
        |    |\  |   \
        a10--R01-a11
         \       R     \
                b00--R10-b01 
           zfac  |  \|   |
                 |---R1--| yfac
               \ |   |   | 
                b10--R11-b11
   */

    // Find all nearest integers surrounding our coordinate
    int Xbelow = static_cast<int>(floor(coord.x));
    int Xabove = static_cast<int>(ceil(coord.x));
    int Ybelow = static_cast<int>(floor(coord.y));
    int Yabove = static_cast<int>(ceil(coord.y));
    int Zbelow = static_cast<int>(floor(coord.z));
    int Zabove = static_cast<int>(ceil(coord.z));

    // Get values of gradient for all surrounding data points
    GradientVoxel a00 = getGradient(Xbelow, Yabove, Zbelow);
    GradientVoxel a01 = getGradient(Xabove, Yabove, Zbelow);
    GradientVoxel a10 = getGradient(Xbelow, Ybelow, Zbelow);
    GradientVoxel a11 = getGradient(Xabove, Ybelow, Zbelow);

    GradientVoxel b00 = getGradient(Xbelow, Yabove, Zabove);
    GradientVoxel b01 = getGradient(Xabove, Yabove, Zabove);
    GradientVoxel b10 = getGradient(Xbelow, Ybelow, Zabove);
    GradientVoxel b11 = getGradient(Xabove, Ybelow, Zabove);

    // Calculate how far along the distance between the surrounding points coord is, in x, y and z direction
    float xFactor = (coord.x - floor(coord.x));
    float yFactor = (coord.y - floor(coord.y));
    float zFactor = (coord.z - floor(coord.z));

    // Use linear interpolation in the x direction for the top and bottom at both z coordinates
    GradientVoxel R00 = linearInterpolate(a00, a01, xFactor);
    GradientVoxel R01 = linearInterpolate(a10, a11, xFactor);
    GradientVoxel R10 = linearInterpolate(b00, b01, xFactor);
    GradientVoxel R11 = linearInterpolate(b10, b11, xFactor);

    // Use linear interpolation between coordinates in a frame and b frame, in y direction
    GradientVoxel R0 = linearInterpolate(R01, R00, yFactor);
    GradientVoxel R1 = linearInterpolate(R11, R10, yFactor);

    // Use linear interpolation in z direction
    return linearInterpolate(R0, R1, zFactor);
}

// ======= TODO : IMPLEMENT ========
// This function should linearly interpolates the value from g0 to g1 given the factor (t).
// At t=0, linearInterpolate should return g0 and at t=1 it returns g1.
GradientVoxel GradientVolume::linearInterpolate(const GradientVoxel& g0, const GradientVoxel& g1, float factor)
{
    // linearly interpolate for the x, y and z directions and magnitude separately
    float xIntpol = (g1.dir.x - g0.dir.x) * factor + g0.dir.x;
    float yIntpol = (g1.dir.y - g0.dir.y) * factor + g0.dir.y;
    float zIntpol = (g1.dir.z - g0.dir.z) * factor + g0.dir.z;
    float magIntpol = (g1.magnitude - g0.magnitude) * factor + g0.magnitude;

    // create direction vector from x, y and z values
    glm::vec3 dirIntpol = glm::vec3(xIntpol, yIntpol, zIntpol);

    // combine into one gradientVoxel object
    return GradientVoxel { dirIntpol, magIntpol };
}

// This function returns a gradientVoxel without using interpolation
GradientVoxel GradientVolume::getGradient(int x, int y, int z) const
{
    const size_t i = static_cast<size_t>(x + m_dim.x * (y + m_dim.y * z));
    return m_data[i];
}
}