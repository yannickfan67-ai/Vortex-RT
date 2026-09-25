#include "vortexrt/buffer.hpp"
#include "vortexrt/build_config.hpp"
#include "vortexrt/compute_pipeline.hpp"
#include "vortexrt/vulkan_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string device;
    std::uint32_t element_count = 8u * 1024u * 1024u;
    std::uint32_t repetitions = 64u;
};

std::uint32_t parse_u32(const std::string& text, const char* name) {
    std::size_t used = 0;
    const unsigned long long value = std::stoull(text, &used, 10);
    if (used != text.size() || value == 0 ||
        value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(name) + " must be a positive uint32");
    }
    return static_cast<std::uint32_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options{};

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--device") {
            options.device = require_value("--device");
        } else if (arg == "--elements") {
            options.element_count =
                parse_u32(require_value("--elements"), "--elements");
        } else if (arg == "--repetitions") {
            options.repetitions =
                parse_u32(require_value("--repetitions"), "--repetitions");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: vortexrt-bench [options]\n"
                << "  --device <index|name>    Vulkan device selector\n"
                << "  --elements <count>       float elements per vector\n"
                << "  --repetitions <count>    dispatch repetitions\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::uint32_t element_count = options.element_count;
        const std::uint32_t repetitions = options.repetitions;
        const VkDeviceSize bytes =
            static_cast<VkDeviceSize>(element_count) * sizeof(float);

        vortexrt::VulkanContext context(options.device);

        std::vector<float> host_a(element_count, 1.25f);
        std::vector<float> host_b(element_count, 2.5f);
        std::vector<float> host_c(element_count, 0.0f);

        vortexrt::Buffer a(
            context,
            bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vortexrt::Buffer b(
            context,
            bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vortexrt::Buffer c(
            context,
            bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        a.upload(host_a.data(), static_cast<std::size_t>(bytes));
        b.upload(host_b.data(), static_cast<std::size_t>(bytes));

        vortexrt::ComputePipeline pipeline(
            context,
            vortexrt::build_config::vector_add_spv,
            3,
            256);

        const std::uint32_t warmup_repetitions =
            std::min<std::uint32_t>(4, repetitions);
        pipeline.run(
            {&a, &b, &c},
            element_count,
            warmup_repetitions);

        const auto stats = pipeline.run(
            {&a, &b, &c},
            element_count,
            repetitions);

        c.download(host_c.data(), static_cast<std::size_t>(bytes));

        const std::size_t validation_stride =
            std::max<std::size_t>(1, host_c.size() / 2048);
        for (std::size_t i = 0; i < host_c.size(); i += validation_stride) {
            if (std::fabs(host_c[i] - 3.75f) > 1e-6f) {
                throw std::runtime_error("vector_add validation failed");
            }
        }

        if (!host_c.empty() &&
            std::fabs(host_c.back() - 3.75f) > 1e-6f) {
            throw std::runtime_error("vector_add tail validation failed");
        }

        std::cout << "Vortex-RT vector_add benchmark\n";
        std::cout << "  Device index: "
                  << context.capabilities().device_index << "\n";
        std::cout << "  GPU: " << context.capabilities().name << "\n";
        std::cout << "  Elements: " << element_count << "\n";
        std::cout << "  Repetitions: " << repetitions << "\n";

        const double logical_bytes =
            static_cast<double>(bytes) * 3.0 * repetitions;

        if (stats.gpu_timing_available && stats.gpu_ms > 0.0) {
            const double seconds = stats.gpu_ms / 1000.0;
            const double gib_per_s =
                logical_bytes / seconds /
                (1024.0 * 1024.0 * 1024.0);

            std::cout << "  GPU time: "
                      << stats.gpu_ms << " ms total\n";
            std::cout << "  Avg dispatch: "
                      << (stats.gpu_ms / repetitions) << " ms\n";
            std::cout << "  Logical kernel bandwidth: "
                      << gib_per_s << " GiB/s\n";
        } else {
            std::cout << "  GPU timestamp timing unavailable\n";
        }

        std::cout << "  CPU submit+wait: "
                  << stats.cpu_ms << " ms total\n";
        std::cout << "  Validation: OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vortexrt-bench: " << e.what() << "\n";
        return 1;
    }
}
