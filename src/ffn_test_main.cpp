#include "vortexrt/build_config.hpp"
#include "vortexrt/buffer.hpp"
#include "vortexrt/q8_matvec.hpp"
#include "vortexrt/vulkan_context.hpp"

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
    const std::uint32_t x =
        std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign =
        (x >> 16) & 0x8000u;
    std::uint32_t mantissa =
        x & 0x7fffffu;
    int exponent =
        static_cast<int>((x >> 23) & 0xffu) -
        127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa =
            (mantissa | 0x800000u) >>
            (1 - exponent);
        return static_cast<std::uint16_t>(
            sign +
            ((mantissa + 0x1000u) >> 13));
    }

    if (exponent >= 31) {
        return static_cast<std::uint16_t>(
            sign | 0x7c00u);
    }

    return static_cast<std::uint16_t>(
        sign |
        (static_cast<std::uint32_t>(exponent) << 10) |
        ((mantissa + 0x1000u) >> 13));
}

float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign =
        std::uint32_t(h & 0x8000u) << 16;
    std::uint32_t exponent =
        (h >> 10) & 31u;
    std::uint32_t mantissa =
        h & 1023u;
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
                (static_cast<std::uint32_t>(e + 127)
                 << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits =
            sign |
            0x7f800000u |
            (mantissa << 13);
    } else {
        bits =
            sign |
            ((exponent - 15u + 127u) << 23) |
            (mantissa << 13);
    }

    return std::bit_cast<float>(bits);
}

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

struct Matrix {
    std::uint32_t offset = 0;
    std::uint32_t input_dim = 0;
    std::uint32_t output_dim = 0;
    std::vector<float> dequantized;
};

Matrix append_q8_matrix(
    std::vector<std::uint8_t>& encoded,
    std::uint32_t input_dim,
    std::uint32_t output_dim,
    std::uint32_t seed) {

    if ((input_dim % 32u) != 0u) {
        throw std::runtime_error(
            "test matrix input must be divisible by 32");
    }

    while ((encoded.size() & 3u) != 0u) {
        encoded.push_back(0);
    }

    Matrix matrix{};
    matrix.offset =
        static_cast<std::uint32_t>(encoded.size());
    matrix.input_dim = input_dim;
    matrix.output_dim = output_dim;
    matrix.dequantized.resize(
        static_cast<std::size_t>(input_dim) *
        output_dim);

    const std::uint32_t blocks =
        input_dim / 32u;

    for (std::uint32_t row = 0;
         row < output_dim;
         ++row) {
        for (std::uint32_t block = 0;
             block < blocks;
             ++block) {
            float values[32]{};
            float abs_max = 0.0f;

            for (std::uint32_t k = 0;
                 k < 32u;
                 ++k) {
                const std::uint32_t col =
                    block * 32u + k;
                const float phase =
                    static_cast<float>(
                        seed * 97u +
                        row * 131u +
                        col * 17u);
                values[k] =
                    std::sin(phase * 0.0031f) *
                    0.45f +
                    std::cos(phase * 0.0017f) *
                    0.15f;
                abs_max =
                    std::max(
                        abs_max,
                        std::fabs(values[k]));
            }

            const float scale =
                abs_max == 0.0f
                    ? 1.0f
                    : abs_max / 127.0f;
            const std::uint16_t half =
                f32_to_f16(scale);
            const float dequant_scale =
                f16_to_f32(half);

            const std::size_t base =
                encoded.size();
            encoded.resize(base + 34u);
            std::memcpy(
                encoded.data() + base,
                &half,
                sizeof(half));

            for (std::uint32_t k = 0;
                 k < 32u;
                 ++k) {
                const int q =
                    std::clamp(
                        static_cast<int>(
                            std::nearbyint(
                                values[k] /
                                dequant_scale)),
                        -127,
                        127);

                encoded[base + 2u + k] =
                    static_cast<std::uint8_t>(
                        static_cast<std::int8_t>(q));

                const std::uint32_t col =
                    block * 32u + k;
                matrix.dequantized[
                    static_cast<std::size_t>(row) *
                        input_dim +
                    col] =
                    dequant_scale *
                    static_cast<float>(q);
            }
        }
    }

    return matrix;
}

std::vector<float> matvec(
    const Matrix& matrix,
    const std::vector<float>& input) {

    if (input.size() != matrix.input_dim) {
        throw std::runtime_error(
            "reference matvec shape mismatch");
    }

    std::vector<float> output(
        matrix.output_dim,
        0.0f);

    for (std::uint32_t row = 0;
         row < matrix.output_dim;
         ++row) {
        double sum = 0.0;
        for (std::uint32_t col = 0;
             col < matrix.input_dim;
             ++col) {
            sum +=
                static_cast<double>(
                    matrix.dequantized[
                        static_cast<std::size_t>(row) *
                            matrix.input_dim +
                        col]) *
                static_cast<double>(input[col]);
        }
        output[row] =
            static_cast<float>(sum);
    }

    return output;
}

} // namespace

int main() {
    try {
        constexpr std::uint32_t kInput = 64;
        constexpr std::uint32_t kFfn = 96;
        constexpr std::uint32_t kOutput = 64;

        std::vector<std::uint8_t> encoded;
        const Matrix gate =
            append_q8_matrix(
                encoded,
                kInput,
                kFfn,
                1);
        const Matrix up =
            append_q8_matrix(
                encoded,
                kInput,
                kFfn,
                2);
        const Matrix down =
            append_q8_matrix(
                encoded,
                kFfn,
                kOutput,
                3);

        while ((encoded.size() & 3u) != 0u) {
            encoded.push_back(0);
        }

        std::vector<float> input(kInput);
        for (std::uint32_t i = 0;
             i < kInput;
             ++i) {
            input[i] =
                std::sin(
                    static_cast<float>(i) *
                    0.071f) *
                0.8f;
        }

        auto gate_ref =
            matvec(gate, input);
        const auto up_ref =
            matvec(up, input);

        for (std::size_t i = 0;
             i < gate_ref.size();
             ++i) {
            gate_ref[i] =
                gelu_tanh(gate_ref[i]) *
                up_ref[i];
        }

        const auto reference =
            matvec(down, gate_ref);

        vortexrt::VulkanContext context;

        vortexrt::Buffer weights(
            context,
            encoded.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vortexrt::Buffer gpu_input(
            context,
            kInput * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vortexrt::Buffer workspace(
            context,
            16384,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vortexrt::Buffer staging_input(
            context,
            kInput * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vortexrt::Buffer staging_output(
            context,
            kOutput * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        weights.upload(
            encoded.data(),
            encoded.size());

        vortexrt::Q8MatVecPipeline pipeline(
            context,
            vortexrt::build_config::q8_matvec_spv,
            vortexrt::build_config::q8_matvec_u8_spv,
            vortexrt::build_config::gelu_mul_spv);

        if (!pipeline.ffn_available()) {
            throw std::runtime_error(
                "fused FFN pipeline was not created");
        }

        std::vector<float> actual(kOutput);
        std::vector<float> cached(kOutput);

        pipeline.run_staged_ffn(
            weights,
            gpu_input,
            workspace,
            staging_input,
            staging_output,
            input.data(),
            input.size() * sizeof(float),
            actual.data(),
            actual.size() * sizeof(float),
            gate.offset,
            up.offset,
            down.offset,
            kInput,
            kFfn,
            kOutput);

        pipeline.run_staged_ffn(
            weights,
            gpu_input,
            workspace,
            staging_input,
            staging_output,
            input.data(),
            input.size() * sizeof(float),
            cached.data(),
            cached.size() * sizeof(float),
            gate.offset,
            up.offset,
            down.offset,
            kInput,
            kFfn,
            kOutput);

        float max_error = 0.0f;
        float cache_error = 0.0f;

        for (std::uint32_t i = 0;
             i < kOutput;
             ++i) {
            max_error =
                std::max(
                    max_error,
                    std::fabs(
                        actual[i] -
                        reference[i]));
            cache_error =
                std::max(
                    cache_error,
                    std::fabs(
                        cached[i] -
                        actual[i]));
        }

        std::cout
            << "Vortex-RT fused FFN\n"
            << "  GPU: "
            << context.capabilities().name
            << "\n"
            << "  Native 8-bit path: "
            << (pipeline.using_native_u8()
                    ? "yes"
                    : "no")
            << "\n"
            << "  max abs error: "
            << max_error
            << "\n"
            << "  cached replay error: "
            << cache_error
            << "\n";

        if (max_error > 5.0e-3f ||
            cache_error > 1.0e-6f) {
            throw std::runtime_error(
                "fused Vulkan FFN mismatch");
        }

        std::cout << "  Validation: OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr
            << "vortexrt-ffn-test: "
            << e.what()
            << "\n";
        return 1;
    }
}
