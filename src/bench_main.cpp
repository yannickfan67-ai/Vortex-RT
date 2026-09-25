#include "vortexrt/buffer.hpp"
#include "vortexrt/build_config.hpp"
#include "vortexrt/compute_pipeline.hpp"
#include "vortexrt/vulkan_context.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        constexpr std::uint32_t element_count = 8u * 1024u * 1024u;
        constexpr std::uint32_t repetitions = 64u;
        constexpr VkDeviceSize bytes =
            static_cast<VkDeviceSize>(element_count) * sizeof(float);

        vortexrt::VulkanContext context;

        std::vector<float> host_a(element_count, 1.25f);
        std::vector<float> host_b(element_count, 2.5f);
        std::vector<float> host_c(element_count, 0.0f);

        vortexrt::Buffer a(
            context,
            bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vortexrt::Buffer b(
            context,
            bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vortexrt::Buffer c(
            context,
            bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        a.upload(host_a.data(), static_cast<std::size_t>(bytes));
        b.upload(host_b.data(), static_cast<std::size_t>(bytes));

        vortexrt::ComputePipeline pipeline(
            context,
            vortexrt::build_config::vector_add_spv,
            3,
            256);

        pipeline.run({&a, &b, &c}, element_count, 4);
        const auto stats = pipeline.run({&a, &b, &c}, element_count, repetitions);

        c.download(host_c.data(), static_cast<std::size_t>(bytes));

        for (std::size_t i = 0; i < host_c.size(); i += 4096) {
            if (std::fabs(host_c[i] - 3.75f) > 1e-6f) {
                throw std::runtime_error("vector_add validation failed");
            }
        }

        std::cout << "Vortex-RT vector_add benchmark\n";
        std::cout << "  GPU: " << context.capabilities().name << "\n";
        std::cout << "  Elements: " << element_count << "\n";
        std::cout << "  Repetitions: " << repetitions << "\n";

        const double transferred_bytes =
            static_cast<double>(bytes) * 3.0 * repetitions;

        if (stats.gpu_timing_available && stats.gpu_ms > 0.0) {
            const double seconds = stats.gpu_ms / 1000.0;
            const double gib_per_s =
                transferred_bytes / seconds / (1024.0 * 1024.0 * 1024.0);

            std::cout << "  GPU time: " << stats.gpu_ms << " ms total\n";
            std::cout << "  Avg dispatch: "
                      << (stats.gpu_ms / repetitions) << " ms\n";
            std::cout << "  Effective bandwidth: "
                      << gib_per_s << " GiB/s\n";
        } else {
            std::cout << "  GPU timestamp timing unavailable\n";
        }

        std::cout << "  CPU submit+wait: " << stats.cpu_ms << " ms total\n";
        std::cout << "  Validation: OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vortexrt-bench: " << e.what() << "\n";
        return 1;
    }
}
