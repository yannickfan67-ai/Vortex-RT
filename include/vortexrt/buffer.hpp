#pragma once

#include "vortexrt/vulkan_context.hpp"

#include <vulkan/vulkan.h>

#include <cstddef>

namespace vortexrt {

class Buffer {
public:
    Buffer(
        VulkanContext& context,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags memory_properties);

    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }

    void upload(const void* data, std::size_t bytes, VkDeviceSize offset = 0);
    void download(void* data, std::size_t bytes, VkDeviceSize offset = 0) const;

private:
    void copy_from(VkBuffer src, VkDeviceSize bytes, VkDeviceSize dst_offset = 0);
    void copy_to(VkBuffer dst, VkDeviceSize bytes, VkDeviceSize src_offset = 0) const;

    VulkanContext& context_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkMemoryPropertyFlags memory_properties_ = 0;
};

} // namespace vortexrt
