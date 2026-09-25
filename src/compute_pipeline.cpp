#include "vortexrt/compute_pipeline.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace vortexrt {
namespace {

void vk_check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(what);
    }
}

std::vector<std::uint32_t> load_spirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open SPIR-V: " + path);
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % 4) != 0) {
        throw std::runtime_error("Invalid SPIR-V size: " + path);
    }

    file.seekg(0, std::ios::beg);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / 4);

    if (!file.read(reinterpret_cast<char*>(words.data()), size)) {
        throw std::runtime_error("Failed to read SPIR-V: " + path);
    }

    return words;
}

} // namespace

ComputePipeline::ComputePipeline(
    VulkanContext& context,
    const std::string& spirv_path,
    std::uint32_t storage_buffer_bindings,
    std::uint32_t local_size_x)
    : context_(context),
      binding_count_(storage_buffer_bindings),
      local_size_x_(local_size_x) {

    if (binding_count_ == 0 || local_size_x_ == 0) {
        throw std::runtime_error("ComputePipeline requires non-zero bindings/local size");
    }

    const auto spirv = load_spirv(spirv_path);

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size() * sizeof(std::uint32_t);
    smci.pCode = spirv.data();

    VkShaderModule shader = VK_NULL_HANDLE;
    vk_check(
        vkCreateShaderModule(context_.device(), &smci, nullptr, &shader),
        "vkCreateShaderModule failed");

    try {
        std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count_);
        for (std::uint32_t i = 0; i < binding_count_; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = binding_count_;
        dlci.pBindings = bindings.data();

        vk_check(
            vkCreateDescriptorSetLayout(
                context_.device(),
                &dlci,
                nullptr,
                &descriptor_set_layout_),
            "vkCreateDescriptorSetLayout failed");

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(std::uint32_t);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &descriptor_set_layout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &push;

        vk_check(
            vkCreatePipelineLayout(
                context_.device(),
                &plci,
                nullptr,
                &pipeline_layout_),
            "vkCreatePipelineLayout failed");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = pipeline_layout_;

        vk_check(
            vkCreateComputePipelines(
                context_.device(),
                VK_NULL_HANDLE,
                1,
                &cpci,
                nullptr,
                &pipeline_),
            "vkCreateComputePipelines failed");

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = binding_count_;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &pool_size;

        vk_check(
            vkCreateDescriptorPool(
                context_.device(),
                &dpci,
                nullptr,
                &descriptor_pool_),
            "vkCreateDescriptorPool failed");

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = descriptor_pool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &descriptor_set_layout_;

        vk_check(
            vkAllocateDescriptorSets(
                context_.device(),
                &dsai,
                &descriptor_set_),
            "vkAllocateDescriptorSets failed");

        VkCommandPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_ci.queueFamilyIndex = context_.compute_queue_family();

        vk_check(
            vkCreateCommandPool(
                context_.device(),
                &pool_ci,
                nullptr,
                &command_pool_),
            "vkCreateCommandPool failed");

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = command_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;

        vk_check(
            vkAllocateCommandBuffers(
                context_.device(),
                &cbai,
                &command_buffer_),
            "vkAllocateCommandBuffers failed");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vk_check(
            vkCreateFence(
                context_.device(),
                &fci,
                nullptr,
                &fence_),
            "vkCreateFence failed");

        if (context_.capabilities().timestamp_valid_bits != 0) {
            VkQueryPoolCreateInfo qpci{};
            qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = 2;
            vk_check(
                vkCreateQueryPool(
                    context_.device(),
                    &qpci,
                    nullptr,
                    &query_pool_),
                "vkCreateQueryPool failed");
        }

        bound_buffers_.assign(binding_count_, VK_NULL_HANDLE);
        bound_sizes_.assign(binding_count_, 0);
    } catch (...) {
        vkDestroyShaderModule(context_.device(), shader, nullptr);
        cleanup();
        throw;
    }

    vkDestroyShaderModule(context_.device(), shader, nullptr);
}

ComputePipeline::~ComputePipeline() {
    cleanup();
}

void ComputePipeline::cleanup() noexcept {
    if (query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(context_.device(), query_pool_, nullptr);
        query_pool_ = VK_NULL_HANDLE;
    }
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(context_.device(), fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
    }
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context_.device(), command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
        command_buffer_ = VK_NULL_HANDLE;
    }
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context_.device(), descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        descriptor_set_ = VK_NULL_HANDLE;
    }
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(context_.device(), pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context_.device(), pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            context_.device(),
            descriptor_set_layout_,
            nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }
}

void ComputePipeline::update_descriptors(
    const std::vector<Buffer*>& buffers) {

    bool changed = false;
    for (std::uint32_t i = 0; i < binding_count_; ++i) {
        if (buffers[i] == nullptr) {
            throw std::runtime_error("ComputePipeline::run received null buffer");
        }
        if (bound_buffers_[i] != buffers[i]->handle() ||
            bound_sizes_[i] != buffers[i]->size()) {
            changed = true;
            break;
        }
    }

    if (!changed) {
        return;
    }

    std::vector<VkDescriptorBufferInfo> infos(binding_count_);
    std::vector<VkWriteDescriptorSet> writes(binding_count_);

    for (std::uint32_t i = 0; i < binding_count_; ++i) {
        infos[i].buffer = buffers[i]->handle();
        infos[i].offset = 0;
        infos[i].range = buffers[i]->size();

        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];

        bound_buffers_[i] = buffers[i]->handle();
        bound_sizes_[i] = buffers[i]->size();
    }

    vkUpdateDescriptorSets(
        context_.device(),
        static_cast<std::uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr);
}

void ComputePipeline::record_commands(
    std::uint32_t element_count,
    std::uint32_t repetitions) {

    if (command_recording_valid_ &&
        recorded_element_count_ == element_count &&
        recorded_repetitions_ == repetitions) {
        return;
    }

    vk_check(
        vkResetCommandBuffer(command_buffer_, 0),
        "vkResetCommandBuffer failed");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vk_check(
        vkBeginCommandBuffer(command_buffer_, &begin),
        "vkBeginCommandBuffer failed");

    if (query_pool_ != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(command_buffer_, query_pool_, 0, 2);
        vkCmdWriteTimestamp(
            command_buffer_,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            query_pool_,
            0);
    }

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

    VkMemoryBarrier before_dispatch{};
    before_dispatch.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before_dispatch.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    before_dispatch.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        command_buffer_,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &before_dispatch,
        0,
        nullptr,
        0,
        nullptr);

    const std::uint32_t groups =
        (element_count + local_size_x_ - 1) / local_size_x_;

    for (std::uint32_t i = 0; i < repetitions; ++i) {
        vkCmdDispatch(command_buffer_, groups, 1, 1);

        if (i + 1 < repetitions) {
            VkMemoryBarrier between_dispatches{};
            between_dispatches.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            between_dispatches.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            between_dispatches.dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT;

            vkCmdPipelineBarrier(
                command_buffer_,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1,
                &between_dispatches,
                0,
                nullptr,
                0,
                nullptr);
        }
    }

    VkMemoryBarrier after_dispatch{};
    after_dispatch.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    after_dispatch.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after_dispatch.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT |
        VK_ACCESS_MEMORY_WRITE_BIT;

    vkCmdPipelineBarrier(
        command_buffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        1,
        &after_dispatch,
        0,
        nullptr,
        0,
        nullptr);

    if (query_pool_ != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(
            command_buffer_,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            query_pool_,
            1);
    }

    vk_check(
        vkEndCommandBuffer(command_buffer_),
        "vkEndCommandBuffer failed");

    recorded_element_count_ = element_count;
    recorded_repetitions_ = repetitions;
    command_recording_valid_ = true;
}

DispatchStats ComputePipeline::run(
    const std::vector<Buffer*>& buffers,
    std::uint32_t element_count,
    std::uint32_t repetitions) {

    if (buffers.size() != binding_count_) {
        throw std::runtime_error("ComputePipeline::run buffer count mismatch");
    }
    if (element_count == 0) {
        throw std::runtime_error("ComputePipeline::run element count must be non-zero");
    }
    if (repetitions == 0) {
        throw std::runtime_error("ComputePipeline::run repetitions must be non-zero");
    }

    update_descriptors(buffers);
    record_commands(element_count, repetitions);

    vk_check(
        vkResetFences(context_.device(), 1, &fence_),
        "vkResetFences failed");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;

    const auto cpu_start = std::chrono::steady_clock::now();

    vk_check(
        vkQueueSubmit(
            context_.compute_queue(),
            1,
            &submit,
            fence_),
        "vkQueueSubmit failed");
    vk_check(
        vkWaitForFences(
            context_.device(),
            1,
            &fence_,
            VK_TRUE,
            UINT64_MAX),
        "vkWaitForFences failed");

    const auto cpu_end = std::chrono::steady_clock::now();

    DispatchStats stats{};
    stats.cpu_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start).count();

    if (query_pool_ != VK_NULL_HANDLE) {
        std::uint64_t ts[2]{};
        vk_check(
            vkGetQueryPoolResults(
                context_.device(),
                query_pool_,
                0,
                2,
                sizeof(ts),
                ts,
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults failed");

        std::uint64_t delta = ts[1] - ts[0];
        const std::uint32_t bits =
            context_.capabilities().timestamp_valid_bits;

        if (bits > 0 && bits < 64) {
            const std::uint64_t mask = (1ull << bits) - 1ull;
            delta &= mask;
        }

        stats.gpu_ms =
            (static_cast<double>(delta) *
             static_cast<double>(
                 context_.capabilities().timestamp_period_ns)) /
            1'000'000.0;
        stats.gpu_timing_available = true;
    }

    return stats;
}

} // namespace vortexrt
