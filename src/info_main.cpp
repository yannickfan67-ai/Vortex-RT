#include "vortexrt/vulkan_context.hpp"

#include <vulkan/vulkan.h>

#include <exception>
#include <iostream>

int main() {
    try {
        vortexrt::VulkanContext context;
        const auto& caps = context.capabilities();

        std::cout << "Vortex-RT Vulkan device\n";
        std::cout << "  GPU: " << caps.name << "\n";
        std::cout << "  Vulkan: "
                  << VK_API_VERSION_MAJOR(caps.api_version) << "."
                  << VK_API_VERSION_MINOR(caps.api_version) << "."
                  << VK_API_VERSION_PATCH(caps.api_version) << "\n";
        std::cout << "  Compute queue family: "
                  << context.compute_queue_family() << "\n";
        std::cout << "  Subgroup size: " << caps.subgroup_size << "\n";
        std::cout << "  FP16 arithmetic: " << (caps.fp16 ? "yes" : "no") << "\n";
        std::cout << "  INT8 arithmetic: " << (caps.int8 ? "yes" : "no") << "\n";
        std::cout << "  8-bit storage: " << (caps.storage8 ? "yes" : "no") << "\n";
        std::cout << "  16-bit storage: " << (caps.storage16 ? "yes" : "no") << "\n";
        std::cout << "  Timestamp bits: " << caps.timestamp_valid_bits << "\n";
        std::cout << "  Timestamp period: " << caps.timestamp_period_ns << " ns\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vortexrt-info: " << e.what() << "\n";
        return 1;
    }
}
