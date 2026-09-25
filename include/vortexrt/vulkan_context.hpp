#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

namespace vortexrt {

struct DeviceCapabilities {
    std::string name;
    std::uint32_t api_version = 0;
    std::uint32_t subgroup_size = 0;
    VkSubgroupFeatureFlags subgroup_ops = 0;

    bool fp16 = false;
    bool int8 = false;
    bool storage8 = false;
    bool storage16 = false;

    float timestamp_period_ns = 0.0f;
    std::uint32_t timestamp_valid_bits = 0;
};

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    [[nodiscard]] VkInstance instance() const noexcept { return instance_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] VkQueue compute_queue() const noexcept { return compute_queue_; }
    [[nodiscard]] std::uint32_t compute_queue_family() const noexcept { return compute_queue_family_; }
    [[nodiscard]] const DeviceCapabilities& capabilities() const noexcept { return caps_; }

    [[nodiscard]] std::uint32_t find_memory_type(
        std::uint32_t type_bits,
        VkMemoryPropertyFlags required) const;

private:
    void create_instance();
    void select_physical_device();
    void create_device();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;

    std::uint32_t compute_queue_family_ = UINT32_MAX;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    DeviceCapabilities caps_{};

    VkPhysicalDeviceVulkan11Features supported11_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features supported12_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
};

} // namespace vortexrt
