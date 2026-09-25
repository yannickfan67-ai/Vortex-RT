#pragma once

#include "vortexrt/buffer.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace vortexrt {

struct DispatchStats {
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
    bool gpu_timing_available = false;
};

class ComputePipeline {
public:
    ComputePipeline(
        VulkanContext& context,
        const std::string& spirv_path,
        std::uint32_t storage_buffer_bindings,
        std::uint32_t local_size_x);

    ~ComputePipeline();

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    DispatchStats run(
        const std::vector<Buffer*>& buffers,
        std::uint32_t element_count,
        std::uint32_t repetitions = 1);

private:
    VulkanContext& context_;
    std::uint32_t binding_count_ = 0;
    std::uint32_t local_size_x_ = 1;

    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
};

} // namespace vortexrt
