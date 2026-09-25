#include "vortexrt/argmax.hpp"

#include <array>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace vortexrt {
namespace {

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(what);
    }
}

std::vector<std::uint32_t> load_spv(
    const std::string& path) {

    std::ifstream file(
        path,
        std::ios::binary |
            std::ios::ate);

    if (!file) {
        throw std::runtime_error(
            "failed to open SPIR-V: " + path);
    }

    const auto size = file.tellg();
    if (size <= 0 ||
        (size % 4) != 0) {
        throw std::runtime_error(
            "invalid SPIR-V: " + path);
    }

    file.seekg(0);

    std::vector<std::uint32_t> words(
        static_cast<std::size_t>(size) / 4);

    if (!file.read(
            reinterpret_cast<char*>(
                words.data()),
            size)) {
        throw std::runtime_error(
            "failed to read SPIR-V: " + path);
    }

    return words;
}

} // namespace

ArgmaxPipeline::ArgmaxPipeline(
    VulkanContext& context,
    const std::string& spirv_path)
    : context_(context) {

    const auto code =
        load_spv(spirv_path);

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize =
        code.size() * sizeof(std::uint32_t);
    shader_info.pCode =
        code.data();

    VkShaderModule shader =
        VK_NULL_HANDLE;

    check(
        vkCreateShaderModule(
            context_.device(),
            &shader_info,
            nullptr,
            &shader),
        "vkCreateShaderModule failed");

    try {
        std::array<
            VkDescriptorSetLayoutBinding,
            2> bindings{};

        for (std::uint32_t i = 0;
             i < bindings.size();
             ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo set_info{};
        set_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_info.bindingCount =
            static_cast<std::uint32_t>(
                bindings.size());
        set_info.pBindings =
            bindings.data();

        check(
            vkCreateDescriptorSetLayout(
                context_.device(),
                &set_info,
                nullptr,
                &set_layout_),
            "vkCreateDescriptorSetLayout failed");

        VkPushConstantRange push{};
        push.stageFlags =
            VK_SHADER_STAGE_COMPUTE_BIT;
        push.size =
            sizeof(std::uint32_t);

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts =
            &set_layout_;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges =
            &push;

        check(
            vkCreatePipelineLayout(
                context_.device(),
                &layout_info,
                nullptr,
                &pipeline_layout_),
            "vkCreatePipelineLayout failed");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage =
            VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module =
            shader;
        stage.pName =
            "main";

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType =
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage =
            stage;
        pipeline_info.layout =
            pipeline_layout_;

        check(
            vkCreateComputePipelines(
                context_.device(),
                VK_NULL_HANDLE,
                1,
                &pipeline_info,
                nullptr,
                &pipeline_),
            "vkCreateComputePipelines failed");

        VkDescriptorPoolSize pool_size{};
        pool_size.type =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 2;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes =
            &pool_size;

        check(
            vkCreateDescriptorPool(
                context_.device(),
                &pool_info,
                nullptr,
                &descriptor_pool_),
            "vkCreateDescriptorPool failed");

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool =
            descriptor_pool_;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts =
            &set_layout_;

        check(
            vkAllocateDescriptorSets(
                context_.device(),
                &alloc_info,
                &descriptor_set_),
            "vkAllocateDescriptorSets failed");

        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.flags =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex =
            context_.compute_queue_family();

        check(
            vkCreateCommandPool(
                context_.device(),
                &command_pool_info,
                nullptr,
                &command_pool_),
            "vkCreateCommandPool failed");

        VkCommandBufferAllocateInfo command_alloc{};
        command_alloc.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_alloc.commandPool =
            command_pool_;
        command_alloc.level =
            VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_alloc.commandBufferCount = 1;

        check(
            vkAllocateCommandBuffers(
                context_.device(),
                &command_alloc,
                &command_buffer_),
            "vkAllocateCommandBuffers failed");

        VkFenceCreateInfo fence_info{};
        fence_info.sType =
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        check(
            vkCreateFence(
                context_.device(),
                &fence_info,
                nullptr,
                &fence_),
            "vkCreateFence failed");
    } catch (...) {
        vkDestroyShaderModule(
            context_.device(),
            shader,
            nullptr);
        cleanup();
        throw;
    }

    vkDestroyShaderModule(
        context_.device(),
        shader,
        nullptr);
}

ArgmaxPipeline::~ArgmaxPipeline() {
    cleanup();
}

void ArgmaxPipeline::cleanup() noexcept {
    const auto device =
        context_.device();

    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(
            device,
            fence_,
            nullptr);
        fence_ =
            VK_NULL_HANDLE;
    }

    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(
            device,
            command_pool_,
            nullptr);
        command_pool_ =
            VK_NULL_HANDLE;
        command_buffer_ =
            VK_NULL_HANDLE;
    }

    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(
            device,
            descriptor_pool_,
            nullptr);
        descriptor_pool_ =
            VK_NULL_HANDLE;
        descriptor_set_ =
            VK_NULL_HANDLE;
    }

    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(
            device,
            pipeline_,
            nullptr);
        pipeline_ =
            VK_NULL_HANDLE;
    }

    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(
            device,
            pipeline_layout_,
            nullptr);
        pipeline_layout_ =
            VK_NULL_HANDLE;
    }

    if (set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            device,
            set_layout_,
            nullptr);
        set_layout_ =
            VK_NULL_HANDLE;
    }
}

void ArgmaxPipeline::update_descriptors(
    Buffer& input,
    Buffer& output) {

    if (bound_input_ == input.handle() &&
        bound_output_ == output.handle() &&
        bound_input_size_ == input.size() &&
        bound_output_size_ == output.size()) {
        return;
    }

    std::array<
        VkDescriptorBufferInfo,
        2> infos{{
            {
                input.handle(),
                0,
                input.size()
            },
            {
                output.handle(),
                0,
                output.size()
            },
        }};

    std::array<
        VkWriteDescriptorSet,
        2> writes{};

    for (std::uint32_t i = 0;
         i < writes.size();
         ++i) {
        writes[i].sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet =
            descriptor_set_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo =
            &infos[i];
    }

    vkUpdateDescriptorSets(
        context_.device(),
        static_cast<std::uint32_t>(
            writes.size()),
        writes.data(),
        0,
        nullptr);

    bound_input_ =
        input.handle();
    bound_output_ =
        output.handle();
    bound_input_size_ =
        input.size();
    bound_output_size_ =
        output.size();
}

void ArgmaxPipeline::record_commands(
    std::uint32_t element_count) {

    if (command_recording_valid_ &&
        recorded_element_count_ ==
            element_count) {
        return;
    }

    check(
        vkResetCommandBuffer(
            command_buffer_,
            0),
        "vkResetCommandBuffer failed");

    VkCommandBufferBeginInfo begin{};
    begin.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags =
        VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    check(
        vkBeginCommandBuffer(
            command_buffer_,
            &begin),
        "vkBeginCommandBuffer failed");

    vkCmdBindPipeline(
        command_buffer_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_);

    vkCmdBindDescriptorSets(
        command_buffer_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout_,
        0,
        1,
        &descriptor_set_,
        0,
        nullptr);

    vkCmdPushConstants(
        command_buffer_,
        pipeline_layout_,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(element_count),
        &element_count);

    VkMemoryBarrier before{};
    before.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before.srcAccessMask =
        VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        command_buffer_,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &before,
        0,
        nullptr,
        0,
        nullptr);

    // The shader intentionally runs one workgroup. Each of its
    // 256 lanes scans a strided slice of the entire input, then
    // reduces the lane maxima in shared memory.
    vkCmdDispatch(
        command_buffer_,
        1,
        1,
        1);

    VkMemoryBarrier after{};
    after.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    after.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT;

    vkCmdPipelineBarrier(
        command_buffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        1,
        &after,
        0,
        nullptr,
        0,
        nullptr);

    check(
        vkEndCommandBuffer(
            command_buffer_),
        "vkEndCommandBuffer failed");

    recorded_element_count_ =
        element_count;
    command_recording_valid_ =
        true;
}

void ArgmaxPipeline::run(
    Buffer& input,
    Buffer& output,
    std::uint32_t element_count) {

    if (element_count == 0) {
        throw std::runtime_error(
            "ArgmaxPipeline requires non-zero input");
    }

    const VkDeviceSize input_bytes =
        static_cast<VkDeviceSize>(
            element_count) *
        sizeof(float);

    if (input_bytes > input.size()) {
        throw std::runtime_error(
            "ArgmaxPipeline input buffer is too small");
    }

    if (sizeof(ArgmaxResult) >
        output.size()) {
        throw std::runtime_error(
            "ArgmaxPipeline output buffer is too small");
    }

    update_descriptors(
        input,
        output);
    record_commands(
        element_count);

    check(
        vkResetFences(
            context_.device(),
            1,
            &fence_),
        "vkResetFences failed");

    VkSubmitInfo submit{};
    submit.sType =
        VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers =
        &command_buffer_;

    check(
        vkQueueSubmit(
            context_.compute_queue(),
            1,
            &submit,
            fence_),
        "vkQueueSubmit failed");

    check(
        vkWaitForFences(
            context_.device(),
            1,
            &fence_,
            VK_TRUE,
            UINT64_MAX),
        "vkWaitForFences failed");
}

} // namespace vortexrt
