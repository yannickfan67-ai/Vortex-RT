#include "vortexrt/build_config.hpp"
#include "vortexrt/q8_matvec.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::uint16_t f32_to_f16(float f) {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::uint32_t mantissa = x & 0x7fffffu;
    int exponent = static_cast<int>((x >> 23) & 0xffu) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        return static_cast<std::uint16_t>(
            sign + ((mantissa + 0x1000u) >> 13));
    }

    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    return static_cast<std::uint16_t>(
        sign |
        (static_cast<std::uint32_t>(exponent) << 10) |
        ((mantissa + 0x1000u) >> 13));
}

float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    std::uint32_t exponent = (h >> 10) & 31u;
    std::uint32_t mantissa = h & 1023u;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x3ffu;
            bits =
                sign |
                (static_cast<std::uint32_t>(e + 127) << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits =
            sign |
            ((exponent - 15u + 127u) << 23) |
            (mantissa << 13);
    }

    return std::bit_cast<float>(bits);
}

} // namespace

int main() {
    try {
        constexpr std::uint32_t kInput = 640;
        constexpr std::uint32_t kOutput = 257;
        constexpr std::uint32_t kBlocks = kInput / 32;

        std::vector<float> input(kInput);
        for (std::uint32_t i = 0; i < kInput; ++i) {
            input[i] = std::sin(float(i) * 0.031f) * 0.75f;
        }

        std::vector<std::uint8_t> quantized(
            std::size_t(kOutput) * kBlocks * 34u);
        std::vector<float> reference(kOutput, 0.0f);

        for (std::uint32_t row = 0; row < kOutput; ++row) {
            for (std::uint32_t block = 0; block < kBlocks; ++block) {
                float values[32];
                float abs_max = 0.0f;

                for (std::uint32_t k = 0; k < 32; ++k) {
                    values[k] =
                        std::cos(
                            float(row * 13 + block * 32 + k) * 0.017f) *
                        0.5f;
                    abs_max = std::max(abs_max, std::fabs(values[k]));
                }

                const float scale = abs_max / 127.0f;
                const std::uint16_t scale_half = f32_to_f16(scale);
                const float dequant_scale = f16_to_f32(scale_half);
                const std::size_t base =
                    (std::size_t(row) * kBlocks + block) * 34u;

                std::memcpy(quantized.data() + base, &scale_half, 2);

                for (std::uint32_t k = 0; k < 32; ++k) {
                    const int q = std::clamp(
                        static_cast<int>(
                            std::nearbyint(values[k] / scale)),
                        -127,
                        127);

                    quantized[base + 2 + k] =
                        static_cast<std::uint8_t>(
                            static_cast<std::int8_t>(q));

                    reference[row] +=
                        dequant_scale *
                        static_cast<float>(q) *
                        input[block * 32 + k];
                }
            }
        }

        while ((quantized.size() & 3u) != 0) {
            quantized.push_back(0);
        }

        vortexrt::VulkanContext context;

        vortexrt::Buffer weights(
            context,
            quantized.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vortexrt::Buffer gpu_input(
            context,
            kInput * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vortexrt::Buffer gpu_output(
            context,
            kOutput * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        weights.upload(quantized.data(), quantized.size());
        gpu_input.upload(
            input.data(),
            input.size() * sizeof(float));

        vortexrt::Q8MatVecPipeline pipeline(
            context,
            vortexrt::build_config::q8_matvec_spv,
            vortexrt::build_config::q8_matvec_u8_spv);

        pipeline.run(
            weights,
            gpu_input,
            gpu_output,
            0,
            kInput,
            kOutput);

        std::vector<float> actual(kOutput);
        gpu_output.download(
            actual.data(),
            actual.size() * sizeof(float));

        float max_error = 0.0f;
        for (std::uint32_t i = 0; i < kOutput; ++i) {
            max_error = std::max(
                max_error,
                std::fabs(actual[i] - reference[i]));
        }

        std::cout << "Vortex-RT Q8_0 matvec\n";
        std::cout << "  GPU: " << context.capabilities().name << "\n";
        std::cout << "  Native 8-bit path: "
                  << (pipeline.using_native_u8() ? "yes" : "no")
                  << "\n";
        std::cout << "  max abs error: " << max_error << "\n";

        if (max_error > 2e-3f) {
            throw std::runtime_error(
                "Q8_0 Vulkan matvec mismatch");
        }

        std::cout << "  Validation: OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vortexrt-q8-test: " << e.what() << "\n";
        return 1;
    }
}
