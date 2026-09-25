#pragma once

#include "vortexrt/buffer.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace vortexrt {

class Q8MatVecPipeline {
public:
    Q8MatVecPipeline(
        VulkanContext& context,
        const std::string& spirv_path,
        const std::string& u8_spirv_path = {},
        const std::string& gelu_mul_spirv_path = {});
    ~Q8MatVecPipeline();

    Q8MatVecPipeline(const Q8MatVecPipeline&) = delete;
    Q8MatVecPipeline& operator=(const Q8MatVecPipeline&) = delete;

    [[nodiscard]] bool using_native_u8() const noexcept {
        return using_native_u8_;
    }

    [[nodiscard]] bool ffn_available() const noexcept {
        return gelu_pipeline_ != VK_NULL_HANDLE;
    }

    void run(
        Buffer& weights,
        Buffer& input,
        Buffer& output,
        std::uint32_t weight_byte_offset,
        std::uint32_t input_dim,
        std::uint32_t output_dim);

    void run_staged(
        Buffer& weights,
        Buffer& input,
        Buffer& output,
        Buffer& staging_input,
        Buffer& staging_output,
        const void* host_input,
        std::size_t host_input_bytes,
        void* host_output,
        std::size_t host_output_bytes,
        std::uint32_t weight_byte_offset,
        std::uint32_t input_dim,
        std::uint32_t output_dim);

    void run_staged_ffn(
        Buffer& weights,
        Buffer& input,
        Buffer& workspace,
        Buffer& staging_input,
        Buffer& staging_output,
        const void* host_input,
        std::size_t host_input_bytes,
        void* host_output,
        std::size_t host_output_bytes,
        std::uint32_t gate_weight_byte_offset,
        std::uint32_t up_weight_byte_offset,
        std::uint32_t down_weight_byte_offset,
        std::uint32_t input_dim,
        std::uint32_t ffn_dim,
        std::uint32_t output_dim);

    void run_staged_pair(
        Buffer& weights,
        Buffer& input,
        Buffer& output,
        Buffer& staging_input,
        Buffer& staging_output,
        const void* host_input,
        std::size_t host_input_bytes,
        void* host_output,
        std::size_t host_output_bytes,
        const std::array<std::uint32_t, 2>& weight_byte_offsets,
        std::uint32_t input_dim,
        const std::array<std::uint32_t, 2>& output_dims);

    void run_staged_triplet(
        Buffer& weights,
        Buffer& input,
        Buffer& output,
        Buffer& staging_input,
        Buffer& staging_output,
        const void* host_input,
        std::size_t host_input_bytes,
        void* host_output,
        std::size_t host_output_bytes,
        const std::array<std::uint32_t, 3>& weight_byte_offsets,
        std::uint32_t input_dim,
        const std::array<std::uint32_t, 3>& output_dims);

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
    bool using_native_u8_ = false;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout gelu_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout gelu_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline gelu_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool ffn_descriptor_pool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 3> ffn_q8_sets_{
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
    };
    VkDescriptorSet ffn_gelu_set_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> pair_sets_{
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
    };
    std::array<VkDescriptorSet, 3> triplet_sets_{
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
    };
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer staged_command_ = VK_NULL_HANDLE;
    VkCommandBuffer triplet_command_ = VK_NULL_HANDLE;
    VkCommandBuffer ffn_command_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    VkBuffer bound_weights_ = VK_NULL_HANDLE;
    VkBuffer bound_input_ = VK_NULL_HANDLE;
    VkBuffer bound_output_ = VK_NULL_HANDLE;
    VkDeviceSize bound_weights_size_ = 0;
    VkDeviceSize bound_input_size_ = 0;
    VkDeviceSize bound_output_size_ = 0;

    VkBuffer pair_bound_weights_ = VK_NULL_HANDLE;
    VkBuffer pair_bound_input_ = VK_NULL_HANDLE;
    VkBuffer pair_bound_output_ = VK_NULL_HANDLE;
    std::array<std::uint32_t, 2> pair_bound_output_dims_{};
    bool pair_descriptors_valid_ = false;

    VkBuffer triplet_bound_weights_ = VK_NULL_HANDLE;
    VkBuffer triplet_bound_input_ = VK_NULL_HANDLE;
    VkBuffer triplet_bound_output_ = VK_NULL_HANDLE;
    std::array<std::uint32_t, 3> triplet_bound_output_dims_{};
    bool triplet_descriptors_valid_ = false;

    VkBuffer ffn_bound_weights_ = VK_NULL_HANDLE;
    VkBuffer ffn_bound_input_ = VK_NULL_HANDLE;
    VkBuffer ffn_bound_workspace_ = VK_NULL_HANDLE;
    std::uint32_t ffn_bound_input_dim_ = 0;
    std::uint32_t ffn_bound_ffn_dim_ = 0;
    std::uint32_t ffn_bound_output_dim_ = 0;
    bool ffn_descriptors_valid_ = false;

    VkDeviceSize storage_buffer_alignment_ = 1;

    std::unordered_map<
        DispatchKey,
        VkCommandBuffer,
        DispatchKeyHash> command_cache_;
};

} // namespace vortexrt
