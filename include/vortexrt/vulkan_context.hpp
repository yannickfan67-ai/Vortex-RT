#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vortexrt {

struct DeviceCapabilities {
    std::string name;
    std::uint32_t device_index = 0;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    VkPhysicalDeviceType device_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    std::uint32_t api_version = 0;
    std::uint32_t subgroup_size = 0;
    VkSubgroupFeatureFlags subgroup_ops = 0;

    bool fp16 = false;
    bool int8 = false;
    bool storage8 = false;
    bool storage16 = false;

    float timestamp_period_ns = 0.0f;
    std::uint32_t timestamp_valid_bits = 0;
    VkDeviceSize non_coherent_atom_size = 1;
    VkDeviceSize device_local_bytes = 0;
};

struct BufferMemoryAllocation {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    std::uint64_t block_id = 0;
    bool pooled = false;
};

class VulkanContext {
public:
    explicit VulkanContext(std::string device_selector = {});
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
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred = 0) const;

    [[nodiscard]] BufferMemoryAllocation allocate_buffer_memory(
        const VkMemoryRequirements& requirements,
        VkMemoryPropertyFlags properties);

    void free_buffer_memory(BufferMemoryAllocation& allocation) noexcept;

    void copy_buffer(
        VkBuffer src,
        VkBuffer dst,
        VkDeviceSize bytes,
        VkDeviceSize src_offset = 0,
        VkDeviceSize dst_offset = 0);

    void stage_upload(
        VkBuffer dst,
        const void* data,
        std::size_t bytes,
        VkDeviceSize dst_offset = 0);

    void stage_download(
        VkBuffer src,
        void* data,
        std::size_t bytes,
        VkDeviceSize src_offset = 0);

private:
    struct FreeRange {
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
    };

    struct MemoryBlock {
        std::uint64_t id = 0;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        std::uint32_t memory_type = 0;
        std::size_t live_allocations = 0;
        std::vector<FreeRange> free_ranges;
    };

    void create_instance();
    void select_physical_device();
    void create_device();
    void create_transfer_resources();
    void destroy_transfer_resources() noexcept;
    void destroy_memory_blocks() noexcept;
    void ensure_staging_capacity(VkDeviceSize bytes);
    void copy_buffer_unlocked(
        VkBuffer src,
        VkBuffer dst,
        VkDeviceSize bytes,
        VkDeviceSize src_offset,
        VkDeviceSize dst_offset);

    std::string device_selector_;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;

    std::uint32_t compute_queue_family_ = UINT32_MAX;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    DeviceCapabilities caps_{};

    VkPhysicalDeviceVulkan11Features supported11_{};
    VkPhysicalDeviceVulkan12Features supported12_{};

    std::mutex memory_mutex_;
    std::vector<MemoryBlock> memory_blocks_;
    std::uint64_t next_memory_block_id_ = 1;

    std::mutex transfer_mutex_;
    VkCommandPool transfer_command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer transfer_command_buffer_ = VK_NULL_HANDLE;
    VkFence transfer_fence_ = VK_NULL_HANDLE;

    VkBuffer staging_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory_ = VK_NULL_HANDLE;
    void* staging_mapped_ = nullptr;
    VkDeviceSize staging_capacity_ = 0;
    VkMemoryPropertyFlags staging_memory_properties_ = 0;
};

} // namespace vortexrt
