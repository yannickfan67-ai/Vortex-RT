#include "vortexrt/vulkan_context.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

namespace vortexrt {
namespace {

void vk_check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(what);
    }
}

std::uint32_t choose_compute_queue(
    VkPhysicalDevice device,
    std::uint32_t& timestamp_valid_bits) {

    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());

    std::uint32_t fallback = UINT32_MAX;

    for (std::uint32_t i = 0; i < count; ++i) {
        const bool compute = (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool graphics = (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;

        if (!compute) {
            continue;
        }

        if (fallback == UINT32_MAX) {
            fallback = i;
        }

        if (!graphics) {
            timestamp_valid_bits = props[i].timestampValidBits;
            return i;
        }
    }

    if (fallback != UINT32_MAX) {
        timestamp_valid_bits = props[fallback].timestampValidBits;
    }

    return fallback;
}

} // namespace

VulkanContext::VulkanContext() {
    create_instance();
    select_physical_device();
    create_device();
}

VulkanContext::~VulkanContext() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void VulkanContext::create_instance() {
    std::uint32_t loader_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) {
        vkEnumerateInstanceVersion(&loader_version);
    }

    if (loader_version < VK_API_VERSION_1_3) {
        throw std::runtime_error("Vortex-RT requires a Vulkan 1.3 loader");
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Vortex-RT";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = "Vortex-RT";
    app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;

    vk_check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance failed");
}

void VulkanContext::select_physical_device() {
    std::uint32_t count = 0;
    vk_check(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
             "vkEnumeratePhysicalDevices failed");

    if (count == 0) {
        throw std::runtime_error("No Vulkan physical device found");
    }

    std::vector<VkPhysicalDevice> devices(count);
    vk_check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
             "vkEnumeratePhysicalDevices failed");

    int best_score = std::numeric_limits<int>::min();

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);

        if (props.apiVersion < VK_API_VERSION_1_3) {
            continue;
        }

        std::uint32_t timestamp_bits = 0;
        const std::uint32_t queue_family = choose_compute_queue(candidate, timestamp_bits);
        if (queue_family == UINT32_MAX) {
            continue;
        }

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 10000;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            score += 5000;
        }

        score += static_cast<int>(props.limits.maxComputeSharedMemorySize / 1024);

        if (score > best_score) {
            best_score = score;
            physical_device_ = candidate;
            compute_queue_family_ = queue_family;
            caps_.timestamp_valid_bits = timestamp_bits;
        }
    }

    if (physical_device_ == VK_NULL_HANDLE) {
        throw std::runtime_error("No Vulkan 1.3 compute-capable GPU found");
    }

    VkPhysicalDeviceSubgroupProperties subgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES
    };

    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical_device_, &props2);

    supported11_ = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    supported12_ = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    supported11_.pNext = &supported12_;

    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &supported11_;
    vkGetPhysicalDeviceFeatures2(physical_device_, &features2);

    caps_.name = props2.properties.deviceName;
    caps_.api_version = props2.properties.apiVersion;
    caps_.subgroup_size = subgroup.subgroupSize;
    caps_.subgroup_ops = subgroup.supportedOperations;
    caps_.fp16 = supported12_.shaderFloat16 == VK_TRUE;
    caps_.int8 = supported12_.shaderInt8 == VK_TRUE;
    caps_.storage8 = supported12_.storageBuffer8BitAccess == VK_TRUE;
    caps_.storage16 = supported11_.storageBuffer16BitAccess == VK_TRUE;
    caps_.timestamp_period_ns = props2.properties.limits.timestampPeriod;

    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);
}

void VulkanContext::create_device() {
    const float priority = 1.0f;

    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = compute_queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan11Features enabled11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES
    };
    enabled11.storageBuffer16BitAccess = supported11_.storageBuffer16BitAccess;

    VkPhysicalDeviceVulkan12Features enabled12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
    };
    enabled12.storageBuffer8BitAccess = supported12_.storageBuffer8BitAccess;
    enabled12.shaderFloat16 = supported12_.shaderFloat16;
    enabled12.shaderInt8 = supported12_.shaderInt8;

    enabled11.pNext = &enabled12;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &enabled11;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    vk_check(vkCreateDevice(physical_device_, &dci, nullptr, &device_),
             "vkCreateDevice failed");

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
}

std::uint32_t VulkanContext::find_memory_type(
    std::uint32_t type_bits,
    VkMemoryPropertyFlags required) const {

    for (std::uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        const bool type_allowed = (type_bits & (1u << i)) != 0;
        const bool flags_match =
            (memory_properties_.memoryTypes[i].propertyFlags & required) == required;

        if (type_allowed && flags_match) {
            return i;
        }
    }

    throw std::runtime_error("No compatible Vulkan memory type found");
}

} // namespace vortexrt
