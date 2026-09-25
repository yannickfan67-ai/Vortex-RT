#include "vortexrt/buffer.hpp"

#include <cstring>
#include <stdexcept>

namespace vortexrt {
namespace {

void vk_check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(what);
    }
}

} // namespace

Buffer::Buffer(
    VulkanContext& context,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memory_properties)
    : context_(context),
      size_(size),
      memory_properties_(memory_properties) {

    if (size_ == 0) {
        throw std::runtime_error("Buffer size must be non-zero");
    }

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size_;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vk_check(
        vkCreateBuffer(context_.device(), &bci, nullptr, &buffer_),
        "vkCreateBuffer failed");

    try {
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(
            context_.device(),
            buffer_,
            &requirements);

        allocation_ = context_.allocate_buffer_memory(
            requirements,
            memory_properties_);

        vk_check(
            vkBindBufferMemory(
                context_.device(),
                buffer_,
                allocation_.memory,
                allocation_.offset),
            "vkBindBufferMemory failed");
    } catch (...) {
        if (allocation_.memory != VK_NULL_HANDLE) {
            context_.free_buffer_memory(allocation_);
        }
        if (buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(context_.device(), buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
        }
        throw;
    }
}

Buffer::~Buffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(context_.device(), buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    if (allocation_.memory != VK_NULL_HANDLE) {
        context_.free_buffer_memory(allocation_);
    }
}

void Buffer::upload(
    const void* data,
    std::size_t bytes,
    VkDeviceSize offset) {

    if (bytes == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::runtime_error("Buffer::upload received null data");
    }
    if (offset > size_ ||
        static_cast<VkDeviceSize>(bytes) > size_ - offset) {
        throw std::runtime_error("Buffer::upload out of range");
    }

    if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        void* mapped = nullptr;
        vk_check(
            vkMapMemory(
                context_.device(),
                allocation_.memory,
                0,
                VK_WHOLE_SIZE,
                0,
                &mapped),
            "vkMapMemory failed");

        auto* dst =
            static_cast<std::byte*>(mapped) +
            allocation_.offset +
            offset;
        std::memcpy(dst, data, bytes);

        if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = allocation_.memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            vk_check(
                vkFlushMappedMemoryRanges(
                    context_.device(),
                    1,
                    &range),
                "vkFlushMappedMemoryRanges failed");
        }

        vkUnmapMemory(context_.device(), allocation_.memory);
        return;
    }

    context_.stage_upload(
        buffer_,
        data,
        bytes,
        offset);
}

void Buffer::download(
    void* data,
    std::size_t bytes,
    VkDeviceSize offset) const {

    if (bytes == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::runtime_error("Buffer::download received null data");
    }
    if (offset > size_ ||
        static_cast<VkDeviceSize>(bytes) > size_ - offset) {
        throw std::runtime_error("Buffer::download out of range");
    }

    if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        void* mapped = nullptr;
        vk_check(
            vkMapMemory(
                context_.device(),
                allocation_.memory,
                0,
                VK_WHOLE_SIZE,
                0,
                &mapped),
            "vkMapMemory failed");

        if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = allocation_.memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            vk_check(
                vkInvalidateMappedMemoryRanges(
                    context_.device(),
                    1,
                    &range),
                "vkInvalidateMappedMemoryRanges failed");
        }

        const auto* src =
            static_cast<const std::byte*>(mapped) +
            allocation_.offset +
            offset;
        std::memcpy(data, src, bytes);

        vkUnmapMemory(context_.device(), allocation_.memory);
        return;
    }

    context_.stage_download(
        buffer_,
        data,
        bytes,
        offset);
}

} // namespace vortexrt
