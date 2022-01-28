#include "volume.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype> // isspace
#include <chrono>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <gsl/span>
#include <iostream>
#include <string>
#include <future> // std::promise, std::future

struct Header {
    glm::ivec3 dim;
    size_t elementSize;
};
static Header readHeader(std::ifstream& ifs);
static float computeMinimum(gsl::span<const uint16_t> data);
static float computeMaximum(gsl::span<const uint16_t> data);
static std::vector<int> computeHistogram(gsl::span<const uint16_t> data);

namespace volume {

Volume::Volume(const std::filesystem::path& file)
    : m_fileName(file.string())
{
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    loadFile(file);
    auto end = clock::now();
    std::cout << "Time to load: " << std::chrono::duration<double, std::milli>(end - start).count() << "ms" << std::endl;

    if (m_data.size() > 0) {
        m_minimum = computeMinimum(m_data);
        m_maximum = computeMaximum(m_data);
        m_histogram = computeHistogram(m_data);
    }
}

Volume::Volume(std::vector<uint16_t> data, const glm::ivec3& dim)
    : m_fileName()
    , m_elementSize(2)
    , m_dim(dim)
    , m_data(std::move(data))
    , m_minimum(computeMinimum(m_data))
    , m_maximum(computeMaximum(m_data))
    , m_histogram(computeHistogram(m_data))
{
}

float Volume::minimum() const
{
    return m_minimum;
}

float Volume::maximum() const
{
    return m_maximum;
}

std::vector<int> Volume::histogram() const
{
    return m_histogram;
}

glm::ivec3 Volume::dims() const
{
    return m_dim;
}

std::string_view Volume::fileName() const
{
    return m_fileName;
}

float Volume::getVoxel(int x, int y, int z) const
{
    const size_t i = size_t(x + m_dim.x * (y + m_dim.y * z));
    return static_cast<float>(m_data[i]);
}

// This function returns a value based on the current interpolation mode
float Volume::getSampleInterpolate(const glm::vec3& coord, float alphaValue) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getSampleNearestNeighbourInterpolation(coord);
    }
    case InterpolationMode::Linear: {
        return getSampleTriLinearInterpolation(coord);
    }
    case InterpolationMode::Cubic: {
        return getSampleTriCubicInterpolation(coord, alphaValue);
    }
    default: {
        throw std::exception();
    }
    }
}

// This function returns the nearest neighbour value at the continuous 3D position given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
float Volume::getSampleNearestNeighbourInterpolation(const glm::vec3& coord) const
{
    // check if the coordinate is within volume boundaries, since we only look at direct neighbours we only need to check within 0.5
    if (glm::any(glm::lessThan(coord + 0.5f, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord + 0.5f, glm::vec3(m_dim))))
        return 0.0f;

    // nearest neighbour simply rounds to the closest voxel positions
    auto roundToPositiveInt = [](float f) {
        // rounding is equal to adding 0.5 and cutting off the fractional part
        return static_cast<int>(f + 0.5f);
    };

    return getVoxel(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// ======= TODO : IMPLEMENT the functions below for tri-linear interpolation ========
// ======= Consider using the linearInterpolate and biLinearInterpolate functions ===
// This function returns the trilinear interpolated value at the continuous 3D position given by coord.
float Volume::getSampleTriLinearInterpolation(const glm::vec3& coord) const
{
    // check if the coordinate is within volume boundaries
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThan(coord, glm::vec3(m_dim.x - 1, m_dim.y - 1, m_dim.z - 1)))) {
        return 0.0f;
    }
    /*
    Tri-linear interpolation example (Variable names based on this art)
            a 
        x00-- --x01
        |    |   | \
        | ---R0--| b 
        |    |\  | 
        x10-- --x11
         \       R     \
              / y00-- --y01 
            c    | \ |   |
                 |---R1--| b 
               \ |   |   | 
                y10-- --y11
   */

    glm::vec2 xyVector = glm::vec2(coord.x, coord.y);

    // Find the z coordinates that surround the point we want to interpolate to
    int z0 = static_cast<int>(floor(coord.z));
    int z1 = static_cast<int>(ceil(coord.z));

    // perform bilinear interpolation for both the front and back frame
    float R0 = biLinearInterpolate(xyVector, z0);
    float R1 = biLinearInterpolate(xyVector, z1);

    float zFactor = (coord.z - floor(coord.z));

    // linearly interpolate the values found in the z direction
    return linearInterpolate(R0, R1, zFactor);
}

// This function linearly interpolates the value at X using incoming values g0 and g1 given a factor (equal to the positon of x in 1D)
// linearInterpolation example (Variable names based on this art)
// g0--X--------g1
//   factor
float Volume::linearInterpolate(float g0, float g1, float factor)
{
    //formula for linear interpolation: x = (g1-g0)*t + g0 
    return (g1 - g0) * factor + g0;
}

// This function bi-linearly interpolates the value at the given continuous 2D XY coordinate for a fixed integer z coordinate.
float Volume::biLinearInterpolate(const glm::vec2& xyCoord, int z) const
{
    /* 
        biLinearInterpolation example (Variable names based on this art)
             a
        x00-- x--x01
        |    |x  |
        | --- -- | b
        |    |   | 
        x10-- x--x11 
                          */

    // Find the datapoints that enclose the point we want to interpolate to
    int Xbelow = static_cast<int>(floor(xyCoord.x));
    int Xabove = static_cast<int>(ceil(xyCoord.x));
    int Ybelow = static_cast<int>(floor(xyCoord.y));
    int Yabove = static_cast<int>(ceil(xyCoord.y));

    // Get voxel values for the surrounding points
    float x00 = getVoxel(Xbelow, Yabove, z);
    float x01 = getVoxel(Xabove, Yabove, z);
    float x10 = getVoxel(Xbelow, Ybelow, z);
    float x11 = getVoxel(Xabove, Ybelow, z);

    // Calculate how far along the distance between the surrounding points x is, in both the x and y direction
    float xFactor = (xyCoord.x - floor(xyCoord.x));
    float yFactor = (xyCoord.y - floor(xyCoord.y));

    // Use 1D interpolation to find the values on the top and bottom axes at the right height
    float xTop = linearInterpolate(x00, x01, xFactor);
    float xBottom = linearInterpolate(x10, x11, xFactor);
    // then interpolate those values vertically to find the final value
    return linearInterpolate(xBottom, xTop, yFactor);
}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This function represents the h(x) function, which returns the weight of the cubic interpolation kernel for a given position x
float Volume::weight(float x, float alphaValue)
{   
    float a = alphaValue; // alphaValue is fetch from UI and can be in range -3.5 to 0 (standard -1)
    float abs_x = abs(x); // Calculate this once to save some computations
    if (abs_x < 1) {
        return (a + 2) * pow(abs_x, 3) - (a + 3) * pow(abs_x, 2) + 1;
    } else if (abs_x < 2) {
        return a * pow(abs_x, 3) - 5 * a * pow(abs_x, 2) + 8 * a * abs_x - 4 * a;
    }
    return 0.0f;
}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This functions returns the results of a cubic interpolation using 4 values and a factor
float Volume::cubicInterpolate(float g0, float g1, float g2, float g3, float factor, float alphaValue)
{   

    // If: factor = dist(g1, X)/dist(g1, g2) we have factor = dist(g1, X),
    // since dist(g1, g2) is 1.
    // g0----g1-X---g2----g3
    return g0 * weight(1 + factor, alphaValue) //Could also be -1 - factor, same since abs() used in weight()
        + g1 * weight(factor, alphaValue) //Could also be -factor, same since abs() used in weight()
        + g2 * weight(1 - factor, alphaValue) 
        + g3 * weight(2 - factor, alphaValue); 
}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This function returns the value of a bicubic interpolation
float Volume::biCubicInterpolate(const glm::vec2& xyCoord, int z, float alphaValue) const
{   
    /*
        biCubicInterpolate example (Variable names based on this art)
        x marks the value, use floor and ceil for corners of 
        closest square and +-1 for further away

        a1100----a1101----a1110---a1111
        |        |        |       |
        |        |        |       |
        |        |        |       |
        a1000----a1001----a1010---a1011
        |        |        |       |
        |        |    x   |       |
        |        |        |       |
        a0100----a0101----a0110---a0111
        |        |        |       |
        |        |        |       |
        |        |        |       |
        a0000----a0001----a0010---a0011 
    
    */

    // Find the datapoints that enclose the point we want to interpolate to
    // For the lowest and highest values also make sure it is inside the dimensions
    int XLowBelow = glm::max(0,static_cast<int>(floor(xyCoord.x))-1);
    int Xbelow = static_cast<int>(floor(xyCoord.x));
    int Xabove = static_cast<int>(ceil(xyCoord.x));
    int XHighabove = glm::min(m_dim.x,static_cast<int>(ceil(xyCoord.x))+1); 

    int YLowBelow = glm::max(0, static_cast<int>(floor(xyCoord.y)) - 1);
    int Ybelow = static_cast<int>(floor(xyCoord.y));
    int Yabove = static_cast<int>(ceil(xyCoord.y));
    int YHighabove = glm::min(m_dim.y, static_cast<int>(ceil(xyCoord.y)) + 1); 

    // Get voxel values for the surrounding points
    float a0000 = getVoxel(XLowBelow, YLowBelow, z);
    float a0001 = getVoxel(Xbelow, YLowBelow, z);
    float a0010 = getVoxel(Xabove, YLowBelow, z);
    float a0011 = getVoxel(XHighabove, YLowBelow, z);

    float a0100 = getVoxel(XLowBelow, Ybelow, z);
    float a0101 = getVoxel(Xbelow, Ybelow, z);
    float a0110 = getVoxel(Xabove, Ybelow, z);
    float a0111 = getVoxel(XHighabove, Ybelow, z);

    float a1000 = getVoxel(XLowBelow, Yabove, z);
    float a1001 = getVoxel(Xbelow, Yabove, z);
    float a1010 = getVoxel(Xabove, Yabove, z);
    float a1011 = getVoxel(XHighabove, Yabove, z);

    float a1100 = getVoxel(XLowBelow, YHighabove, z);
    float a1101 = getVoxel(Xbelow, YHighabove, z);
    float a1110 = getVoxel(Xabove, YHighabove, z);
    float a1111 = getVoxel(XHighabove, YHighabove, z);


    // Calculate how far along the distance between the surrounding points the value is, in both the x and y direction
    // This will return only the values after the decimal point: 2.5 --> 0.5 
    float xFactor = (xyCoord.x - floor(xyCoord.x));
    float yFactor = (xyCoord.y - floor(xyCoord.y));

    // Use 1D interpolation to find the values on the top and bottom axes at the right height
    float lowBottom = cubicInterpolate(a0000, a0001, a0010, a0011, xFactor, alphaValue);
    float bottom = cubicInterpolate(a0100, a0101, a0110, a0111, xFactor, alphaValue);
    float above = cubicInterpolate(a1000, a1001, a1010, a1011, xFactor, alphaValue);
    float highabove = cubicInterpolate(a1100, a1101, a1110, a1111, xFactor, alphaValue);

    // then interpolate those values vertically to find the final value
    return cubicInterpolate(lowBottom, bottom, above, highabove, yFactor, alphaValue);
}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This function computes the tricubic interpolation at coord
/* float Volume::getSampleTriCubicInterpolation(const glm::vec3& coord, float alphaValue) const
{
    // check if the coordinate is within volume boundaries. Need to check 1.5 since we have two neighbours
    if (glm::any(glm::lessThan(coord - 1.5f, glm::vec3(0))) || glm::any(glm::greaterThan(coord + 1.5f, glm::vec3(m_dim - 1)))) {
        // If we check at the outskirts of the dimension with no neighbour  we can still linear interpolat and
        // if outside dimensions linear interpolate will return 0.0f
        return getSampleTriLinearInterpolation(coord);
    }


    // Find the z coordinates that surround the point we want to interpolate to

    std::array<int, 4> zcoordinates = {
        glm::max(0, static_cast<int>(floor(coord.z)) - 1),
        static_cast<int>(floor(coord.z)),
        static_cast<int>(ceil(coord.z)),
        glm::min(m_dim.z, static_cast<int>(ceil(coord.z)) + 1)
    };

    glm::vec2 xyVector = glm::vec2(coord.x, coord.y);

    // perform biCubic interpolation for z-values in front and back of the point

    std::array<std::future<float>, 4> valueArray;

    for (int i = 0; i < zcoordinates.size(); i++) {
        valueArray[i] = std::async(std::launch::async, [this, xyVector, zcoordinates, i, alphaValue]() { return biCubicInterpolate(xyVector, zcoordinates[i], alphaValue); });
    }

    std::array<float, 4> taskedValues;

    for (int i = 0; i < 4; i++)
    {
        taskedValues[i] = valueArray[i].get();
    }

    // Calculate how far along the distance between the surrounding points the value is, in both the x and y direction
    // This will return only the values after the decimal point: 2.5 --> 0.5
    float zFactor = (coord.z - floor(coord.z));

    // cubic interpolate the values found in the z direction
    // retrun 0.0f if the value is negative as an error check
    return glm::max(cubicInterpolate(taskedValues[0], taskedValues[1], taskedValues[2], taskedValues[3], zFactor, alphaValue), 0.0f);
}*/

 float Volume::getSampleTriCubicInterpolation(const glm::vec3& coord, float alphaValue) const
{
    // check if the coordinate is within volume boundaries. Need to check 1.5 since we have two neighbours
    if (glm::any(glm::lessThan(coord - 1.5f, glm::vec3(0))) || glm::any(glm::greaterThan(coord + 1.5f, glm::vec3(m_dim-1)))) {
        // If we check at the outskirts of the dimension with no neighbour  we can still linear interpolat and
        // if outside dimensions linear interpolate will return 0.0f
        return getSampleTriLinearInterpolation(coord);
    }

    // Find the z coordinates that surround the point we want to interpolate to

    int ZLowBelow = glm::max(0, static_cast<int>(floor(coord.z)) - 1);
    int Zbelow = static_cast<int>(floor(coord.z));
    int Zabove = static_cast<int>(ceil(coord.z));
    int ZHighabove = glm::min(m_dim.z, static_cast<int>(ceil(coord.z)) + 1);

    glm::vec2 xyVector = glm::vec2(coord.x, coord.y);

    // perform biCubic interpolation for z-values in front and back of the point


    float C0 = biCubicInterpolate(xyVector, ZLowBelow, alphaValue);
    float C1 = biCubicInterpolate(xyVector, Zbelow, alphaValue);
    float C2 = biCubicInterpolate(xyVector, Zabove, alphaValue);
    float C3 = biCubicInterpolate(xyVector, ZHighabove, alphaValue);

    // Calculate how far along the distance between the surrounding points the value is, in both the x and y direction
    // This will return only the values after the decimal point: 2.5 --> 0.5 
    float zFactor = (coord.z - floor(coord.z));

    // cubic interpolate the values found in the z direction
    // retrun 0.0f if the value is negative as an error check
    return glm::max(cubicInterpolate(C0, C1, C2, C3, zFactor, alphaValue), 0.0f);
}   


// Load an fld volume data file
// First read and parse the header, then the volume data can be directly converted from bytes to uint16_ts
void Volume::loadFile(const std::filesystem::path& file)
{
    assert(std::filesystem::exists(file));
    std::ifstream ifs(file, std::ios::binary);
    assert(ifs.is_open());

    const auto header = readHeader(ifs);
    m_dim = header.dim;
    m_elementSize = header.elementSize;

    const size_t voxelCount = static_cast<size_t>(header.dim.x * header.dim.y * header.dim.z);
    const size_t byteCount = voxelCount * header.elementSize;
    std::vector<char> buffer(byteCount);
    // Data section is separated from header by two /f characters.
    ifs.seekg(2, std::ios::cur);
    ifs.read(buffer.data(), std::streamsize(byteCount));

    m_data.resize(voxelCount);
    if (header.elementSize == 1) { // Bytes.
        for (size_t i = 0; i < byteCount; i++) {
            m_data[i] = static_cast<uint16_t>(buffer[i] & 0xFF);
        }
    } else if (header.elementSize == 2) { // uint16_ts.
        for (size_t i = 0; i < byteCount; i += 2) {
            m_data[i / 2] = static_cast<uint16_t>((buffer[i] & 0xFF) + (buffer[i + 1] & 0xFF) * 256);
        }
    }
}
}

static Header readHeader(std::ifstream& ifs)
{
    Header out {};

    // Read input until the data section starts.
    std::string line;
    while (ifs.peek() != '\f' && !ifs.eof()) {
        std::getline(ifs, line);
        // Remove comments.
        line = line.substr(0, line.find('#'));
        // Remove any spaces from the string.
        // https://stackoverflow.com/questions/83439/remove-spaces-from-stdstring-in-c
        line.erase(std::remove_if(std::begin(line), std::end(line), ::isspace), std::end(line));
        if (line.empty())
            continue;

        const auto separator = line.find('=');
        const auto key = line.substr(0, separator);
        const auto value = line.substr(separator + 1);

        if (key == "ndim") {
            if (std::stoi(value) != 3) {
                std::cout << "Only 3D files supported\n";
            }
        } else if (key == "dim1") {
            out.dim.x = std::stoi(value);
        } else if (key == "dim2") {
            out.dim.y = std::stoi(value);
        } else if (key == "dim3") {
            out.dim.z = std::stoi(value);
        } else if (key == "nspace") {
        } else if (key == "veclen") {
            if (std::stoi(value) != 1)
                std::cerr << "Only scalar m_data are supported" << std::endl;
        } else if (key == "data") {
            if (value == "byte") {
                out.elementSize = 1;
            } else if (value == "short") {
                out.elementSize = 2;
            } else {
                std::cerr << "Data type " << value << " not recognized" << std::endl;
            }
        } else if (key == "field") {
            if (value != "uniform")
                std::cerr << "Only uniform m_data are supported" << std::endl;
        } else if (key == "#") {
            // Comment.
        } else {
            std::cerr << "Invalid AVS keyword " << key << " in file" << std::endl;
        }
    }
    return out;
}

static float computeMinimum(gsl::span<const uint16_t> data)
{
    return float(*std::min_element(std::begin(data), std::end(data)));
}

static float computeMaximum(gsl::span<const uint16_t> data)
{
    return float(*std::max_element(std::begin(data), std::end(data)));
}

static std::vector<int> computeHistogram(gsl::span<const uint16_t> data)
{
    std::vector<int> histogram(size_t(*std::max_element(std::begin(data), std::end(data)) + 1), 0);
    for (const auto v : data)
        histogram[v]++;
    return histogram;
}