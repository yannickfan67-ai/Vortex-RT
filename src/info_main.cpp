#include "vortexrt/vulkan_context.hpp"

#include <vulkan/vulkan.h>

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string parse_device_selector(int argc, char** argv) {
    std::string selector;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--device requires an index or name substring");
            }
            selector = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: vortexrt-info [--device <index|name>]\n"
                << "       VORTEXRT_DEVICE may also select a device.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return selector;
}

} // namespace

int main(int argc, char** argv) {
    try {
        vortexrt::VulkanContext context(parse_device_selector(argc, argv));
        const auto& caps = context.capabilities();

        std::cout << "Vortex-RT Vulkan device\n";
        std::cout << "  Device index: " << caps.device_index << "\n";
        std::cout << "  GPU: " << caps.name << "\n";
        std::cout << "  Vendor ID: 0x"
                  << std::hex << caps.vendor_id << std::dec << "\n";
        std::cout << "  Device ID: 0x"
                  << std::hex << caps.device_id << std::dec << "\n";
        std::cout << "  Vulkan: "
                  << VK_API_VERSION_MAJOR(caps.api_version) << "."
                  << VK_API_VERSION_MINOR(caps.api_version) << "."
                  << VK_API_VERSION_PATCH(caps.api_version) << "\n";
        std::cout << "  Compute queue family: "
                  << context.compute_queue_family() << "\n";
        std::cout << "  Device-local memory: "
                  << (static_cast<double>(caps.device_local_bytes) /
                      (1024.0 * 1024.0 * 1024.0))
                  << " GiB\n";
        std::cout << "  Subgroup size: " << caps.subgroup_size << "\n";
        std::cout << "  FP16 arithmetic: "
                  << (caps.fp16 ? "yes" : "no") << "\n";
        std::cout << "  INT8 arithmetic: "
                  << (caps.int8 ? "yes" : "no") << "\n";
        std::cout << "  8-bit storage: "
                  << (caps.storage8 ? "yes" : "no") << "\n";
        std::cout << "  16-bit storage: "
                  << (caps.storage16 ? "yes" : "no") << "\n";
        std::cout << "  Timestamp bits: "
                  << caps.timestamp_valid_bits << "\n";
        std::cout << "  Timestamp period: "
                  << caps.timestamp_period_ns << " ns\n";
        std::cout << "  Non-coherent atom size: "
                  << caps.non_coherent_atom_size << " bytes\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vortexrt-info: " << e.what() << "\n";
        return 1;
    }
}
