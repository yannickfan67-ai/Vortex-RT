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

void one_shot_copy(
    VulkanContext& context,
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize bytes,
    VkDeviceSize src_offset,
    VkDeviceSize dst_offset) {

    VkCommandPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_ci.queueFamilyIndex = context.compute_queue_family();

    VkCommandPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateCommandPool(context.device(), &pool_ci, nullptr, &pool),
             "vkCreateCommandPool failed");

    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(context.device(), &alloc, &cmd),
             "vkAllocateCommandBuffers failed");

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer failed");

    VkBufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = bytes;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);

    vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer failed");

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    vk_check(vkQueueSubmit(context.compute_queue(), 1, &submit, VK_NULL_HANDLE),
             "vkQueueSubmit failed");
    vk_check(vkQueueWaitIdle(context.compute_queue()), "vkQueueWaitIdle failed");

    vkDestroyCommandPool(context.device(), pool, nullptr);
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

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vk_check(vkCreateBuffer(context_.device(), &bci, nullptr, &buffer_),
             "vkCreateBuffer failed");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(context_.device(), buffer_, &req);

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = context_.find_memory_type(req.memoryTypeBits, memory_properties_);

    try {
        vk_check(vkAllocateMemory(context_.device(), &mai, nullptr, &memory_),
                 "vkAllocateMemory failed");
        vk_check(vkBindBufferMemory(context_.device(), buffer_, memory_, 0),
                 "vkBindBufferMemory failed");
    } catch (...) {
        if (memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(context_.device(), memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
        }
        vkDestroyBuffer(context_.device(), buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        throw;
    }
}

Buffer::~Buffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(context_.device(), buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(context_.device(), memory_, nullptr);
    }
}

void Buffer::upload(const void* data, std::size_t bytes, VkDeviceSize offset) {
    if (offset + bytes > size_) {
        throw std::runtime_error("Buffer::upload out of range");
    }

    if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        void* mapped = nullptr;
        vk_check(vkMapMemory(context_.device(), memory_, offset, bytes, 0, &mapped),
                 "vkMapMemory failed");
        std::memcpy(mapped, data, bytes);

        if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = memory_;
            range.offset = offset;
            range.size = bytes;
            vk_check(vkFlushMappedMemoryRanges(context_.device(), 1, &range),
                     "vkFlushMappedMemoryRanges failed");
        }

        vkUnmapMemory(context_.device(), memory_);
        return;
    }

    Buffer staging(
        context_,
        bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    staging.upload(data, bytes);
    copy_from(staging.handle(), bytes, offset);
}

void Buffer::download(void* data, std::size_t bytes, VkDeviceSize offset) const {
    if (offset + bytes > size_) {
        throw std::runtime_error("Buffer::download out of range");
    }

    if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        void* mapped = nullptr;
        vk_check(vkMapMemory(context_.device(), memory_, offset, bytes, 0, &mapped),
                 "vkMapMemory failed");

        if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = memory_;
            range.offset = offset;
            range.size = bytes;
            vk_check(vkInvalidateMappedMemoryRanges(context_.device(), 1, &range),
                     "vkInvalidateMappedMemoryRanges failed");
        }

        std::memcpy(data, mapped, bytes);
        vkUnmapMemory(context_.device(), memory_);
        return;
    }

    Buffer staging(
        context_,
        bytes,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    copy_to(staging.handle(), bytes, offset);
    staging.download(data, bytes);
}

void Buffer::copy_from(VkBuffer src, VkDeviceSize bytes, VkDeviceSize dst_offset) {
    one_shot_copy(context_, src, buffer_, bytes, 0, dst_offset);
}

void Buffer::copy_to(VkBuffer dst, VkDeviceSize bytes, VkDeviceSize src_offset) const {
    one_shot_copy(context_, buffer_, dst, bytes, src_offset, 0);
}

} // namespace vortexrt
