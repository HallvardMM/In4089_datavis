﻿#include "renderer.h"
#include <algorithm>
#include <algorithm> // std::fill
#include <cmath>
#include <functional>
#include <glm/common.hpp>
#include <glm/gtx/component_wise.hpp>
#include <iostream>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tuple>

namespace render {

// The renderer is passed a pointer to the volume, gradinet volume, camera and an initial renderConfig.
// The camera being pointed to may change each frame (when the user interacts). When the renderConfig
// changes the setConfig function is called with the updated render config. This gives the Renderer an
// opportunity to resize the framebuffer.
Renderer::Renderer(
    const volume::Volume* pVolume,
    const volume::GradientVolume* pGradientVolume,
    const render::RayTraceCamera* pCamera,
    const RenderConfig& initialConfig)
    : m_pVolume(pVolume)
    , m_pGradientVolume(pGradientVolume)
    , m_pCamera(pCamera)
    , m_config(initialConfig)
{
    resizeImage(initialConfig.renderResolution);
}

// Set a new render config if the user changed the settings.
void Renderer::setConfig(const RenderConfig& config)
{
    if (config.renderResolution != m_config.renderResolution)
        resizeImage(config.renderResolution);

    m_config = config;
}

// Resize the framebuffer and fill it with black pixels.
void Renderer::resizeImage(const glm::ivec2& resolution)
{
    m_frameBuffer.resize(size_t(resolution.x) * size_t(resolution.y), glm::vec4(0.0f));
}

// Clear the framebuffer by setting all pixels to black.
void Renderer::resetImage()
{
    std::fill(std::begin(m_frameBuffer), std::end(m_frameBuffer), glm::vec4(0.0f));
}

// Return a VIEW into the framebuffer. This view is merely a reference to the m_frameBuffer member variable.
// This does NOT make a copy of the framebuffer.
gsl::span<const glm::vec4> Renderer::frameBuffer() const
{
    return m_frameBuffer;
}

// Main render function. It computes an image according to the current renderMode.
// Multithreading is enabled in Release/RelWithDebInfo modes. In Debug mode multithreading is disabled to make debugging easier.
void Renderer::render()
{
    resetImage();

    static constexpr float sampleStep = 1.0f;
    const glm::vec3 planeNormal = -glm::normalize(m_pCamera->forward());
    const glm::vec3 volumeCenter = glm::vec3(m_pVolume->dims()) / 2.0f;
    const Bounds bounds { glm::vec3(0.0f), glm::vec3(m_pVolume->dims() - glm::ivec3(1)) };

    // 0 = sequential (single-core), 1 = TBB (multi-core)
#ifdef NDEBUG
    // If NOT in debug mode then enable parallelism using the TBB library (Intel Threaded Building Blocks).
#define PARALLELISM 1
#else
    // Disable multi threading in debug mode.
#define PARALLELISM 0
#endif

#if PARALLELISM == 0
    // Regular (single threaded) for loops.
    for (int x = 0; x < m_config.renderResolution.x; x++) {
        for (int y = 0; y < m_config.renderResolution.y; y++) {
#else
    // Parallel for loop (in 2 dimensions) that subdivides the screen into tiles.
    const tbb::blocked_range2d<int> screenRange { 0, m_config.renderResolution.y, 0, m_config.renderResolution.x };
        tbb::parallel_for(screenRange, [&](tbb::blocked_range2d<int> localRange) {
        // Loop over the pixels in a tile. This function is called on multiple threads at the same time.
        for (int y = std::begin(localRange.rows()); y != std::end(localRange.rows()); y++) {
            for (int x = std::begin(localRange.cols()); x != std::end(localRange.cols()); x++) {
#endif
            // Compute a ray for the current pixel.
            const glm::vec2 pixelPos = glm::vec2(x, y) / glm::vec2(m_config.renderResolution);
            Ray ray = m_pCamera->generateRay(pixelPos * 2.0f - 1.0f);

            // Compute where the ray enters and exists the volume.
            // If the ray misses the volume then we continue to the next pixel.
            if (!instersectRayVolumeBounds(ray, bounds))
                continue;

            // Get a color for the current pixel according to the current render mode.
            glm::vec4 color {};
            switch (m_config.renderMode) {
            case RenderMode::RenderSlicer: {
                color = traceRaySlice(ray, volumeCenter, planeNormal);
                break;
            }
            case RenderMode::RenderMIP: {
                color = traceRayMIP(ray, sampleStep);
                break;
            }
            case RenderMode::RenderComposite: {
                color = traceRayComposite(ray, sampleStep);
                break;
            }
            case RenderMode::RenderIso: {
                color = traceRayISO(ray, sampleStep);
                break;
            }
            case RenderMode::RenderTF2D: {
                color = traceRayTF2D(ray, sampleStep);
                break;
            }
            };
            // Write the resulting color to the screen.
            fillColor(x, y, color);

#if PARALLELISM == 1
        }
    }
});
#else
            }
        }
#endif
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// This function generates a view alongside a plane perpendicular to the camera through the center of the volume
//  using the slicing technique.
glm::vec4 Renderer::traceRaySlice(const Ray& ray, const glm::vec3& volumeCenter, const glm::vec3& planeNormal) const
{
    const float t = glm::dot(volumeCenter - ray.origin, planeNormal) / glm::dot(ray.direction, planeNormal);
    const glm::vec3 samplePos = ray.origin + ray.direction * t;
    const float val = m_pVolume->getSampleInterpolate(samplePos, m_config.alphaTriValue);
    return glm::vec4(glm::vec3(std::max(val / m_pVolume->maximum(), 0.0f)), 1.f);
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Function that implements maximum-intensity-projection (MIP) raycasting.
// It returns the color assigned to a ray/pixel given it's origin, direction and the distances
// at which it enters/exits the volume (ray.tmin & ray.tmax respectively).
// The ray must be sampled with a distance defined by the sampleStep
glm::vec4 Renderer::traceRayMIP(const Ray& ray, float sampleStep) const
{
    float maxVal = 0.0f;

    // Incrementing samplePos directly instead of recomputing it each frame gives a measureable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos, m_config.alphaTriValue);
        maxVal = std::max(val, maxVal);
    }

    // Normalize the result to a range of [0 to mpVolume->maximum()].
    return glm::vec4(glm::vec3(maxVal) / m_pVolume->maximum(), 1.0f);
}

// ======= TODO: IMPLEMENT ========
// This function should find the position where the ray intersects with the volume's isosurface.
// If volume shading is DISABLED then simply return the isoColor.
// If volume shading is ENABLED then return the phong-shaded color at that location using the local gradient (from m_pGradientVolume).
//   Use the camera position (m_pCamera->position()) as the light position.
// Use the bisectionAccuracy function (to be implemented) to get a more precise isosurface location between two steps.
glm::vec4 Renderer::traceRayISO(const Ray& ray, float sampleStep) const
{
    float isoVal = m_config.isoValue;

    // Boolean indicating if, in the current ray, we have already encountered the isovalue
    bool foundISO = false;

    static constexpr glm::vec3 isoColor { 1.0f, 1.0f, 0.2f };
    glm::vec3 color = { 1.0f, 1.0f, 1.0f };

    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;

    // Step through the ray, at each position checking whether we have exceeded the isovalue
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos, m_config.alphaTriValue);
        // We only enter this loop once, the first time we find a value exceeding the isovalue.
        if (val > isoVal && !foundISO) {
            float accurateT = bisectionAccuracy(ray, (t - sampleStep), t, isoVal);
            glm::vec3 accuratePos = ray.origin + accurateT * ray.direction;
            color = computePhongShading(isoColor, m_pGradientVolume->getGradientInterpolate(accuratePos), m_pCamera->position(), ray.direction, m_config.k_a, m_config.k_d, m_config.k_s, m_config.phongAlpha);
            foundISO = true;
            break; // Break to improve performance and not iterate through all values
        }
    }

    // Check if volume shading is switched on and return the correct color.
    // If we have not encountered the isovalue in a ray we return see-through black.
    if (m_config.volumeShading && foundISO) {
        return glm::vec4(color, 1.0f);
    } else if (foundISO) {
        return glm::vec4(isoColor, 1.0f);
    }
    return glm::vec4 { 0.0f };
}

// ======= TODO: IMPLEMENT ========
// Given that the iso value lies somewhere between t0 and t1, find a t for which the value
// closely matches the iso value (less than 0.01 difference). Add a limit to the number of
// iterations such that it does not get stuck in degerate cases.
float Renderer::bisectionAccuracy(const Ray& ray, float t0, float t1, float isoValue) const
{
    float lowerBound = t0;
    float upperBound = t1;
    float lowerValue = m_pVolume->getSampleInterpolate(ray.origin + t0 * ray.direction, m_config.alphaTriValue);
    float upperValue = m_pVolume->getSampleInterpolate(ray.origin + t1 * ray.direction, m_config.alphaTriValue);
    float currentPoint = lowerBound + ((upperBound - lowerBound) / 2);

    for (int i = 0; i <= 200; i += 1) {
        // take point midway between known boundaries
        currentPoint = lowerBound + ((upperBound - lowerBound) / 2);
        const float currentVal = lowerValue + ((upperValue - lowerValue) / 2);

        if (currentVal <= isoValue) {
            if ((isoValue - currentVal) < 0.01f) {
                //We're close enough, return current point
                return currentPoint;
            }
            // We're lower than the value we're looking for, so we need to look above our current point
            // So set the lower bound to our current point and start the loop again
            lowerBound = currentPoint;
            lowerValue = currentVal;
        } else {
            if ((currentVal - isoValue) < 0.01f) {
                //We're close enough, return current point
                return currentPoint;
            }
            // We're higher than the value we're looking for, so look below current point
            // Set upper bound to current point and start the loop again
            upperBound = currentPoint;
            upperValue = currentVal;
        }
    }

    return currentPoint;
}

// ======= TODO: IMPLEMENT ========
// Compute Phong Shading given the voxel color (material color), the gradient, the light vector and view vector.
// You can find out more about the Phong shading model at:
// https://en.wikipedia.org/wiki/Phong_reflection_model
//
// Use the given color for the ambient/specular/diffuse (you are allowed to scale these constants by a scalar value).
// You are free to choose any specular power that you'd like.
glm::vec3 Renderer::computePhongShading(const glm::vec3& color, const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V, float k_a, float k_d,float k_s,float phongAlpha)
{
    // Calculate the inverse of the light direction
    glm::vec3 Linv = glm::vec3(L.x * -1.0f, L.y * -1.0f, L.z * -1.0f);

    // Theta is the angle between the surface normal and the light direction
    float cos_theta = glm::dot(glm::normalize(gradient.dir), glm::normalize(Linv));

    // reflection of L = L−2(L⋅n)n where n is surface normal (normalized)
    // source: https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    glm::vec3 reflectL = L - (2 * glm::dot(L, glm::normalize(gradient.dir)) * glm::normalize(gradient.dir));

    // Phi is the angle between the light reflection and the viewing direction
    float cos_phi = glm::dot(glm::normalize(reflectL), glm::normalize(V));

    // Calculate the three types of light separately, then add together to return
    // k_a, k_d, k_s are the phong weights exposed in the UI. 
    // PhongAlpha is the specular reflection term.
    glm::vec3 ambient = k_a * color;
    glm::vec3 diffuse = k_d * cos_theta * color;
    glm::vec3 specular = glm::vec3(k_s * color.x * pow(cos_phi, phongAlpha), k_s * color.y * pow(cos_phi, phongAlpha), k_s * color.z * pow(cos_phi, phongAlpha));

    return ambient + diffuse + specular;
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 1D transfer function raycasting.
// Use getTFValue to compute the color for a given volume value according to the 1D transfer function.
glm::vec4 Renderer::traceRayComposite(const Ray& ray, float sampleStep) const
{
    // Initialize color and opacity values
    glm::vec4 Cprev = glm::vec4 { 0.0f };
    float Aprev = 0.0f;
    glm::vec4 C_i = glm::vec4 { 0.0f };
    float A_i = 0.0f;

    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;

    // Step through the ray front-to-back as defined on slide 26 of lecture 8
    // calculate the color for each step using phong shading
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos, m_config.alphaTriValue);
        glm::vec4 TFval = getTFValue(val);
        // Scale the colour vector from c_i to C_i by multiplying R, G and B with A
        TFval = glm::vec4(TFval.x * TFval.w, TFval.y * TFval.w, TFval.z * TFval.w, TFval.w);
        glm::vec3 TFcolor = glm::vec3(TFval.x, TFval.y, TFval.z);
        if (TFval.w > 0) {
            glm::vec3 TFcolor = computePhongShading(TFcolor, m_pGradientVolume->getGradientInterpolate(samplePos), m_pCamera->position(), ray.direction, m_config.k_a, m_config.k_d, m_config.k_s, m_config.phongAlpha);
            float w = TFval.w;
            glm::vec4 TFval = glm::vec4(TFcolor.x, TFcolor.y, TFcolor.z, w);
        }

        // The last C_i and A_i now become C_i-1 and A_i-1, and we calculate the next step in the ray traversal.
        Cprev = C_i;
        Aprev = A_i;
        C_i = Cprev + (1 - Aprev) * TFval;
        A_i = Aprev + (1 - Aprev) * TFval.w;

        // If opacity is nearly 1 we break to save computations, as objects behind opaque objects will not be visible
        if (A_i > 0.99) {
            return C_i;
        }
    }
    return C_i;
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Looks up the color+opacity corresponding to the given volume value from the 1D tranfer function LUT (m_config.tfColorMap).
// The value will initially range from (m_config.tfColorMapIndexStart) to (m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) .
glm::vec4 Renderer::getTFValue(float val) const
{
    // Map value from [m_config.tfColorMapIndexStart, m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) to [0, 1) .
    const float range01 = (val - m_config.tfColorMapIndexStart) / m_config.tfColorMapIndexRange;
    const size_t i = std::min(static_cast<size_t>(range01 * static_cast<float>(m_config.tfColorMap.size())), m_config.tfColorMap.size() - 1);
    return m_config.tfColorMap[i];
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 2D transfer function raycasting.
// Use the getTF2DOpacity function that you implemented to compute the opacity according to the 2D transfer function.
glm::vec4 Renderer::traceRayTF2D(const Ray& ray, float sampleStep) const
{
    glm::vec4 Cprev = glm::vec4 { 0.0f };
    float Aprev = 0.0f;
    glm::vec4 C_i = glm::vec4 { 0.0f };
    float A_i = 0.0f;

    // We use front-to-back compositing as defined on slide 26 of lecture 8
    // stepping through the ray and computing C_i at every point
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos, m_config.alphaTriValue);
        // Use the color defined in the colorpicker in the UI
        glm::vec3 TFcolor = m_config.TF2DColor;
        volume::GradientVoxel gradient = m_pGradientVolume->getGradientInterpolate(samplePos);

        float TFopacity = getTF2DOpacity(val, gradient.magnitude);
        // Scale the color from c_i to C_i by multiplying RGB with opacity
        glm::vec4 TFval = glm::vec4(TFcolor.x * TFopacity, TFcolor.y * TFopacity, TFcolor.z * TFopacity, TFopacity);

        // The last C_i and A_i now become C_i-1 and A_i-1, and we calculate the next step in the ray traversal.
        Cprev = C_i;
        Aprev = A_i;
        C_i = Cprev + (1 - Aprev) * TFval;
        A_i = Aprev + (1 - Aprev) * TFopacity;

        // If opacity is nearly 1 we break to save computations, as objects behind opaque objects will not be visible (early ray termination)
        if (A_i > 0.99) {
            return C_i;
        }
    }
    return C_i;
}

// ======= TODO: IMPLEMENT ========
// This function should return an opacity value for the given intensity and gradient according to the 2D transfer function.
// Calculate whether the values are within the radius/intensity triangle defined in the 2D transfer function widget.
// If so: return a tent weighting as described in the assignment
// Otherwise: return 0.0f
//
// The 2D transfer function settings can be accessed through m_config.TF2DIntensity and m_config.TF2DRadius.
float Renderer::getTF2DOpacity(float intensity, float gradientMagnitude) const
{
    // Get the transfer function intensity and radius
    float TFintensity = m_config.TF2DIntensity;
    float TFradius = m_config.TF2DRadius;

    // Get the maximum gradient magnitude
    float maxMagnitude = m_pGradientVolume->maxMagnitude();

    // Calculate the distance between the transfer function intensity and the triangle side, given a gradient magnitude
    float r = abs(gradientMagnitude * TFradius / maxMagnitude);

    // If the given intensity and gradient magnitude falls within the given area, return the opacity as given by the normalised distance
    // from the sample intensity to the edge of the area, otherwise return opacity 0.
    if (abs(intensity - TFintensity) <= r) {
        return (r - abs(intensity - TFintensity)) / r;
    }

    return 0.0f;
}

// This function computes if a ray intersects with the axis-aligned bounding box around the volume.
// If the ray intersects then tmin/tmax are set to the distance at which the ray hits/exists the
// volume and true is returned. If the ray misses the volume the the function returns false.
//
// If you are interested you can learn about it at.
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
bool Renderer::instersectRayVolumeBounds(Ray& ray, const Bounds& bounds) const
{
    const glm::vec3 invDir = 1.0f / ray.direction;
    const glm::bvec3 sign = glm::lessThan(invDir, glm::vec3(0.0f));

    float tmin = (bounds.lowerUpper[sign[0]].x - ray.origin.x) * invDir.x;
    float tmax = (bounds.lowerUpper[!sign[0]].x - ray.origin.x) * invDir.x;
    const float tymin = (bounds.lowerUpper[sign[1]].y - ray.origin.y) * invDir.y;
    const float tymax = (bounds.lowerUpper[!sign[1]].y - ray.origin.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    tmin = std::max(tmin, tymin);
    tmax = std::min(tmax, tymax);

    const float tzmin = (bounds.lowerUpper[sign[2]].z - ray.origin.z) * invDir.z;
    const float tzmax = (bounds.lowerUpper[!sign[2]].z - ray.origin.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    ray.tmin = std::max(tmin, tzmin);
    ray.tmax = std::min(tmax, tzmax);
    return true;
}

// This function inserts a color into the framebuffer at position x,y
void Renderer::fillColor(int x, int y, const glm::vec4& color)
{
    const size_t index = static_cast<size_t>(m_config.renderResolution.x * y + x);
    m_frameBuffer[index] = color;
}
}