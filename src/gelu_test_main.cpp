#include "vortexrt/build_config.hpp"
#include "vortexrt/buffer.hpp"
#include "vortexrt/compute_pipeline.hpp"
#include "vortexrt/vulkan_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi =
        0.7978845608028654f;
    constexpr float kCubic =
        0.044715f;

    const float x3 = x * x * x;
    return 0.5f * x *
        (1.0f +
         std::tanh(
             kSqrt2OverPi *
             (x + kCubic * x3)));
}

} // namespace

int main() {
    try {
        constexpr std::uint32_t kElements = 4096;

        std::vector<float> gate(kElements);
        std::vector<float> up(kElements);
        std::vector<float> reference(kElements);

        for (std::uint32_t i = 0;
             i < kElements;
             ++i) {
            gate[i] =
                std::sin(
                    static_cast<float>(i) *
                    0.0137f) *
                5.0f;
            up[i] =
                std::cos(
                    static_cast<float>(i) *
                    0.0211f) *
                3.0f;
            reference[i] =
                gelu_tanh(gate[i]) *
                up[i];
        }

        vortexrt::VulkanContext context;

        vortexrt::Buffer gate_buffer(
            context,
            gate.size() * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vortexrt::Buffer up_buffer(
            context,
            up.size() * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        gate_buffer.upload(
            gate.data(),
            gate.size() * sizeof(float));
        up_buffer.upload(
            up.data(),
            up.size() * sizeof(float));

        vortexrt::ComputePipeline pipeline(
            context,
            vortexrt::build_config::gelu_mul_spv,
            2,
            256);

        const auto stats =
            pipeline.run(
                {
                    &gate_buffer,
                    &up_buffer,
                },
                kElements,
                1);

        std::vector<float> actual(kElements);
        gate_buffer.download(
            actual.data(),
            actual.size() * sizeof(float));

        float max_error = 0.0f;
        for (std::uint32_t i = 0;
             i < kElements;
             ++i) {
            max_error =
                std::max(
                    max_error,
                    std::fabs(
                        actual[i] -
                        reference[i]));
        }

        std::cout
            << "Vortex-RT GELU gate-up kernel\n"
            << "  GPU: "
            << context.capabilities().name
            << "\n"
            << "  Elements: "
            << kElements
            << "\n"
            << "  max abs error: "
            << max_error
            << "\n"
            << "  dispatch CPU time: "
            << stats.cpu_ms
            << " ms\n";

        if (stats.gpu_timing_available) {
            std::cout
                << "  dispatch GPU time: "
                << stats.gpu_ms
                << " ms\n";
        }

        if (max_error > 5.0e-5f) {
            throw std::runtime_error(
                "Vulkan GELU gate-up kernel mismatch");
        }

        std::cout << "  Validation: OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr
            << "vortexrt-gelu-test: "
            << e.what()
            << "\n";
        return 1;
    }
}
