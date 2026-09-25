#pragma once

#include "vortexrt/buffer.hpp"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

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
    struct DispatchKey {
        std::uint32_t weight_byte_offset = 0;
        std::uint32_t input_dim = 0;
        std::uint32_t output_dim = 0;

        bool operator==(const DispatchKey&) const noexcept = default;
    };

    struct DispatchKeyHash {
        std::size_t operator()(const DispatchKey& key) const noexcept;
    };

    void cleanup() noexcept;

    [[nodiscard]] VkCommandBuffer get_or_record_command(
        const DispatchKey& key);

    VulkanContext& context_;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    VkBuffer bound_weights_ = VK_NULL_HANDLE;
    VkBuffer bound_input_ = VK_NULL_HANDLE;
    VkBuffer bound_output_ = VK_NULL_HANDLE;
    VkDeviceSize bound_weights_size_ = 0;
    VkDeviceSize bound_input_size_ = 0;
    VkDeviceSize bound_output_size_ = 0;

    std::unordered_map<
        DispatchKey,
        VkCommandBuffer,
        DispatchKeyHash> command_cache_;
};

} // namespace vortexrt
