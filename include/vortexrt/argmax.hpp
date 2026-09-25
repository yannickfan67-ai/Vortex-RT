#pragma once

#include "vortexrt/buffer.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

namespace vortexrt {

struct ArgmaxResult {
    std::uint32_t index = 0;
    float value = 0.0f;
};

static_assert(sizeof(ArgmaxResult) == 8);

class ArgmaxPipeline {
public:
    ArgmaxPipeline(
        VulkanContext& context,
        const std::string& spirv_path);
    ~ArgmaxPipeline();

    ArgmaxPipeline(const ArgmaxPipeline&) = delete;
    ArgmaxPipeline& operator=(const ArgmaxPipeline&) = delete;

    void run(
        Buffer& input,
        Buffer& output,
        std::uint32_t element_count);

private:
    void cleanup() noexcept;
    void update_descriptors(
        Buffer& input,
        Buffer& output);
    void record_commands(
        std::uint32_t element_count);

    VulkanContext& context_;

    VkDescriptorSetLayout set_layout_ =
        VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ =
        VK_NULL_HANDLE;
    VkPipeline pipeline_ =
        VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ =
        VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ =
        VK_NULL_HANDLE;
    VkCommandPool command_pool_ =
        VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ =
        VK_NULL_HANDLE;
    VkFence fence_ =
        VK_NULL_HANDLE;

    VkBuffer bound_input_ = VK_NULL_HANDLE;
    VkBuffer bound_output_ = VK_NULL_HANDLE;
    VkDeviceSize bound_input_size_ = 0;
    VkDeviceSize bound_output_size_ = 0;
    std::uint32_t recorded_element_count_ = 0;
    bool command_recording_valid_ = false;
};

} // namespace vortexrt
