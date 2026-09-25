#pragma once

#include "vortexrt/buffer.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

namespace vortexrt {

class Q8MatVecPipeline {
public:
    Q8MatVecPipeline(
        VulkanContext& context,
        const std::string& spirv_path);
    ~Q8MatVecPipeline();

    Q8MatVecPipeline(const Q8MatVecPipeline&) = delete;
    Q8MatVecPipeline& operator=(const Q8MatVecPipeline&) = delete;

    void run(
        Buffer& weights,
        Buffer& input,
        Buffer& output,
        std::uint32_t weight_byte_offset,
        std::uint32_t input_dim,
        std::uint32_t output_dim);

private:
    void cleanup() noexcept;

    VulkanContext& context_;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    VkBuffer bound_weights_ = VK_NULL_HANDLE;
    VkBuffer bound_input_ = VK_NULL_HANDLE;
    VkBuffer bound_output_ = VK_NULL_HANDLE;
    VkDeviceSize bound_weights_size_ = 0;
    VkDeviceSize bound_input_size_ = 0;
    VkDeviceSize bound_output_size_ = 0;

    std::uint32_t recorded_weight_byte_offset_ = 0;
    std::uint32_t recorded_input_dim_ = 0;
    std::uint32_t recorded_output_dim_ = 0;
    bool command_recording_valid_ = false;
};

} // namespace vortexrt
