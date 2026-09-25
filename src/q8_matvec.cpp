#include "vortexrt/q8_matvec.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vortexrt {
namespace {

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(what);
    }
}

std::vector<std::uint32_t> load_spv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("failed to open SPIR-V: " + path);
    }

    const auto size = file.tellg();
    if (size <= 0 || (size % 4) != 0) {
        throw std::runtime_error("invalid SPIR-V");
    }

    file.seekg(0);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / 4);
    if (!file.read(reinterpret_cast<char*>(words.data()), size)) {
        throw std::runtime_error("failed to read SPIR-V");
    }
    return words;
}

struct Push {
    std::uint32_t offset;
    std::uint32_t in_dim;
    std::uint32_t out_dim;
    std::uint32_t groups_x;
};

constexpr std::uint64_t kQ8BlockElements = 32;
constexpr std::uint64_t kQ8BlockBytes = 34;

std::uint64_t checked_mul(
    std::uint64_t a,
    std::uint64_t b,
    const char* what) {

    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::runtime_error(what);
    }
    return a * b;
}

std::uint64_t checked_add(
    std::uint64_t a,
    std::uint64_t b,
    const char* what) {

    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::runtime_error(what);
    }
    return a + b;
}

} // namespace

Q8MatVecPipeline::Q8MatVecPipeline(
    VulkanContext& context,
    const std::string& path,
    const std::string& u8_path)
    : context_(context) {

    using_native_u8_ =
        !u8_path.empty() &&
        context.capabilities().storage8 &&
        context.capabilities().int8;

    const auto code =
        load_spv(
            using_native_u8_
                ? u8_path
                : path);

    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = code.size() * sizeof(std::uint32_t);
    sm.pCode = code.data();

    VkShaderModule shader = VK_NULL_HANDLE;
    check(
        vkCreateShaderModule(context.device(), &sm, nullptr, &shader),
        "vkCreateShaderModule failed");

    try {
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo sl{};
        sl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        sl.bindingCount = static_cast<std::uint32_t>(bindings.size());
        sl.pBindings = bindings.data();
        check(
            vkCreateDescriptorSetLayout(
                context.device(), &sl, nullptr, &set_layout_),
            "vkCreateDescriptorSetLayout failed");

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.size = sizeof(Push);

        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &set_layout_;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &range;
        check(
            vkCreatePipelineLayout(
                context.device(), &pl, nullptr, &pipeline_layout_),
            "vkCreatePipelineLayout failed");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp.stage = stage;
        cp.layout = pipeline_layout_;
        check(
            vkCreateComputePipelines(
                context.device(),
                VK_NULL_HANDLE,
                1,
                &cp,
                nullptr,
                &pipeline_),
            "vkCreateComputePipelines failed");

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 12;

        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 4;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &ps;
        check(
            vkCreateDescriptorPool(
                context.device(), &dp, nullptr, &descriptor_pool_),
            "vkCreateDescriptorPool failed");

        std::array<VkDescriptorSetLayout, 4> layouts{
            set_layout_,
            set_layout_,
            set_layout_,
            set_layout_,
        };
        std::array<VkDescriptorSet, 4> sets{};

        VkDescriptorSetAllocateInfo da{};
        da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        da.descriptorPool = descriptor_pool_;
        da.descriptorSetCount =
            static_cast<std::uint32_t>(sets.size());
        da.pSetLayouts = layouts.data();
        check(
            vkAllocateDescriptorSets(
                context.device(),
                &da,
                sets.data()),
            "vkAllocateDescriptorSets failed");

        descriptor_set_ = sets[0];
        triplet_sets_[0] = sets[1];
        triplet_sets_[1] = sets[2];
        triplet_sets_[2] = sets[3];

        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = context.compute_queue_family();
        check(
            vkCreateCommandPool(
                context.device(), &pci, nullptr, &command_pool_),
            "vkCreateCommandPool failed");

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check(
            vkCreateFence(
                context.device(), &fi, nullptr, &fence_),
            "vkCreateFence failed");
    } catch (...) {
        vkDestroyShaderModule(context.device(), shader, nullptr);
        cleanup();
        throw;
    }

    vkDestroyShaderModule(context.device(), shader, nullptr);
}

Q8MatVecPipeline::~Q8MatVecPipeline() {
    cleanup();
}

void Q8MatVecPipeline::cleanup() noexcept {
    const auto device = context_.device();

    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device, fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
    }
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
        command_cache_.clear();
    }
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        descriptor_set_ = VK_NULL_HANDLE;
        triplet_sets_.fill(VK_NULL_HANDLE);
    }
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
        set_layout_ = VK_NULL_HANDLE;
    }
}

std::size_t Q8MatVecPipeline::DispatchKeyHash::operator()(
    const DispatchKey& key) const noexcept {

    std::size_t hash =
        static_cast<std::size_t>(key.weight_byte_offset);

    const auto combine = [&](std::uint32_t value) {
        hash ^=
            static_cast<std::size_t>(value) +
            static_cast<std::size_t>(0x9e3779b9u) +
            (hash << 6u) +
            (hash >> 2u);
    };

    combine(key.input_dim);
    combine(key.output_dim);
    return hash;
}

VkCommandBuffer Q8MatVecPipeline::get_or_record_command(
    const DispatchKey& key) {

    if (const auto it = command_cache_.find(key);
        it != command_cache_.end()) {
        return it->second;
    }

    constexpr std::uint32_t kMaxGroupsX = 65535u;
    constexpr std::uint32_t kMaxGroupsY = 65535u;

    const std::uint32_t groups_x =
        std::min(key.output_dim, kMaxGroupsX);
    const std::uint64_t groups_y_64 =
        (static_cast<std::uint64_t>(key.output_dim) +
         groups_x - 1u) /
        groups_x;

    if (groups_y_64 > kMaxGroupsY) {
        throw std::runtime_error(
            "Q8MatVecPipeline output dimension exceeds 2D dispatch capacity");
    }

    VkCommandBuffer command = VK_NULL_HANDLE;

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = command_pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    check(
        vkAllocateCommandBuffers(
            context_.device(),
            &ai,
            &command),
        "vkAllocateCommandBuffers failed");

    try {
        VkCommandBufferBeginInfo begin{};
        begin.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags =
            VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

        check(
            vkBeginCommandBuffer(command, &begin),
            "vkBeginCommandBuffer failed");

        vkCmdBindPipeline(
            command,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_);
        vkCmdBindDescriptorSets(
            command,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_,
            0,
            1,
            &descriptor_set_,
            0,
            nullptr);

        const Push push{
            key.weight_byte_offset,
            key.input_dim,
            key.output_dim,
            groups_x,
        };

        vkCmdPushConstants(
            command,
            pipeline_layout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(push),
            &push);

        VkMemoryBarrier before_dispatch{};
        before_dispatch.sType =
            VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        before_dispatch.srcAccessMask =
            VK_ACCESS_MEMORY_WRITE_BIT;
        before_dispatch.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &before_dispatch,
            0,
            nullptr,
            0,
            nullptr);

        vkCmdDispatch(
            command,
            groups_x,
            static_cast<std::uint32_t>(groups_y_64),
            1);

        VkMemoryBarrier after_dispatch{};
        after_dispatch.sType =
            VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after_dispatch.srcAccessMask =
            VK_ACCESS_SHADER_WRITE_BIT;
        after_dispatch.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT |
            VK_ACCESS_MEMORY_WRITE_BIT;

        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            1,
            &after_dispatch,
            0,
            nullptr,
            0,
            nullptr);

        check(
            vkEndCommandBuffer(command),
            "vkEndCommandBuffer failed");

        command_cache_.emplace(key, command);
        return command;
    } catch (...) {
        if (command != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(
                context_.device(),
                command_pool_,
                1,
                &command);
        }
        throw;
    }
}

void Q8MatVecPipeline::run(
    Buffer& weights,
    Buffer& input,
    Buffer& output,
    std::uint32_t weight_byte_offset,
    std::uint32_t input_dim,
    std::uint32_t output_dim) {

    if (input_dim == 0 || output_dim == 0) {
        throw std::runtime_error(
            "Q8MatVecPipeline dimensions must be non-zero");
    }
    if ((input_dim % kQ8BlockElements) != 0) {
        throw std::runtime_error(
            "Q8MatVecPipeline input_dim must be divisible by 32");
    }
    if ((weight_byte_offset & 3u) != 0u) {
        throw std::runtime_error(
            "Q8MatVecPipeline weight byte offset must be 4-byte aligned");
    }

    const std::uint64_t blocks =
        static_cast<std::uint64_t>(input_dim) / kQ8BlockElements;
    const std::uint64_t row_bytes =
        checked_mul(
            blocks,
            kQ8BlockBytes,
            "Q8MatVecPipeline row byte size overflow");
    const std::uint64_t matrix_bytes =
        checked_mul(
            static_cast<std::uint64_t>(output_dim),
            row_bytes,
            "Q8MatVecPipeline matrix byte size overflow");
    const std::uint64_t required_weight_bytes =
        checked_add(
            weight_byte_offset,
            matrix_bytes,
            "Q8MatVecPipeline weight range overflow");

    const std::uint64_t required_input_bytes =
        checked_mul(
            input_dim,
            sizeof(float),
            "Q8MatVecPipeline input byte size overflow");
    const std::uint64_t required_output_bytes =
        checked_mul(
            output_dim,
            sizeof(float),
            "Q8MatVecPipeline output byte size overflow");

    if (required_weight_bytes > weights.size()) {
        throw std::runtime_error(
            "Q8MatVecPipeline weight buffer is too small");
    }
    if (required_input_bytes > input.size()) {
        throw std::runtime_error(
            "Q8MatVecPipeline input buffer is too small");
    }
    if (required_output_bytes > output.size()) {
        throw std::runtime_error(
            "Q8MatVecPipeline output buffer is too small");
    }

    const bool descriptor_changed =
        bound_weights_ != weights.handle() ||
        bound_input_ != input.handle() ||
        bound_output_ != output.handle() ||
        bound_weights_size_ != weights.size() ||
        bound_input_size_ != input.size() ||
        bound_output_size_ != output.size();

    if (descriptor_changed) {
        std::array<VkDescriptorBufferInfo, 3> infos{{
            {weights.handle(), 0, weights.size()},
            {input.handle(), 0, input.size()},
            {output.handle(), 0, output.size()},
        }};

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t i = 0; i < writes.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set_;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }

        vkUpdateDescriptorSets(
            context_.device(),
            static_cast<std::uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);

        bound_weights_ = weights.handle();
        bound_input_ = input.handle();
        bound_output_ = output.handle();
        bound_weights_size_ = weights.size();
        bound_input_size_ = input.size();
        bound_output_size_ = output.size();
    }

    check(
        vkResetFences(context_.device(), 1, &fence_),
        "vkResetFences failed");

    const DispatchKey dispatch_key{
        weight_byte_offset,
        input_dim,
        output_dim,
    };
    const VkCommandBuffer command =
        get_or_record_command(dispatch_key);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;

    check(
        vkQueueSubmit(
            context_.compute_queue(), 1, &submit, fence_),
        "vkQueueSubmit failed");
    check(
        vkWaitForFences(
            context_.device(), 1, &fence_, VK_TRUE, UINT64_MAX),
        "vkWaitForFences failed");
}


void Q8MatVecPipeline::run_staged(
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
    std::uint32_t output_dim) {

    if (host_input == nullptr) {
        throw std::runtime_error(
            "Q8MatVecPipeline::run_staged received null host input");
    }
    const bool readback = host_output != nullptr;
    if (input_dim == 0 || output_dim == 0) {
        throw std::runtime_error(
            "Q8MatVecPipeline dimensions must be non-zero");
    }
    if ((input_dim % kQ8BlockElements) != 0) {
        throw std::runtime_error(
            "Q8MatVecPipeline input_dim must be divisible by 32");
    }
    if ((weight_byte_offset & 3u) != 0u) {
        throw std::runtime_error(
            "Q8MatVecPipeline weight byte offset must be 4-byte aligned");
    }

    const std::uint64_t blocks =
        static_cast<std::uint64_t>(input_dim) /
        kQ8BlockElements;
    const std::uint64_t row_bytes =
        checked_mul(
            blocks,
            kQ8BlockBytes,
            "Q8MatVecPipeline row byte size overflow");
    const std::uint64_t matrix_bytes =
        checked_mul(
            static_cast<std::uint64_t>(output_dim),
            row_bytes,
            "Q8MatVecPipeline matrix byte size overflow");
    const std::uint64_t required_weight_bytes =
        checked_add(
            weight_byte_offset,
            matrix_bytes,
            "Q8MatVecPipeline weight range overflow");
    const std::uint64_t required_input_bytes =
        checked_mul(
            input_dim,
            sizeof(float),
            "Q8MatVecPipeline input byte size overflow");
    const std::uint64_t required_output_bytes =
        checked_mul(
            output_dim,
            sizeof(float),
            "Q8MatVecPipeline output byte size overflow");

    if (required_weight_bytes > weights.size() ||
        required_input_bytes > input.size() ||
        required_output_bytes > output.size()) {
        throw std::runtime_error(
            "Q8MatVecPipeline device buffer is too small");
    }
    if (required_input_bytes > staging_input.size() ||
        (readback &&
         required_output_bytes > staging_output.size())) {
        throw std::runtime_error(
            "Q8MatVecPipeline staging buffer is too small");
    }
    if (host_input_bytes < required_input_bytes ||
        (readback &&
         host_output_bytes < required_output_bytes)) {
        throw std::runtime_error(
            "Q8MatVecPipeline host buffer is too small");
    }

    staging_input.upload(
        host_input,
        static_cast<std::size_t>(required_input_bytes));

    const bool descriptor_changed =
        bound_weights_ != weights.handle() ||
        bound_input_ != input.handle() ||
        bound_output_ != output.handle() ||
        bound_weights_size_ != weights.size() ||
        bound_input_size_ != input.size() ||
        bound_output_size_ != output.size();

    if (descriptor_changed) {
        std::array<VkDescriptorBufferInfo, 3> infos{{
            {weights.handle(), 0, weights.size()},
            {input.handle(), 0, input.size()},
            {output.handle(), 0, output.size()},
        }};

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t i = 0; i < writes.size(); ++i) {
            writes[i].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set_;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }

        vkUpdateDescriptorSets(
            context_.device(),
            static_cast<std::uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);

        bound_weights_ = weights.handle();
        bound_input_ = input.handle();
        bound_output_ = output.handle();
        bound_weights_size_ = weights.size();
        bound_input_size_ = input.size();
        bound_output_size_ = output.size();
    }

    if (staged_command_ == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = command_pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        check(
            vkAllocateCommandBuffers(
                context_.device(),
                &ai,
                &staged_command_),
            "vkAllocateCommandBuffers failed for staged Q8 path");
    }

    check(
        vkResetCommandBuffer(staged_command_, 0),
        "vkResetCommandBuffer failed for staged Q8 path");

    VkCommandBufferBeginInfo begin{};
    begin.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(
        vkBeginCommandBuffer(staged_command_, &begin),
        "vkBeginCommandBuffer failed for staged Q8 path");

    VkBufferCopy input_copy{};
    input_copy.size =
        static_cast<VkDeviceSize>(required_input_bytes);
    vkCmdCopyBuffer(
        staged_command_,
        staging_input.handle(),
        input.handle(),
        1,
        &input_copy);

    VkMemoryBarrier input_barrier{};
    input_barrier.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT;
    input_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        staged_command_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &input_barrier,
        0,
        nullptr,
        0,
        nullptr);

    vkCmdBindPipeline(
        staged_command_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_);
    vkCmdBindDescriptorSets(
        staged_command_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout_,
        0,
        1,
        &descriptor_set_,
        0,
        nullptr);

    constexpr std::uint32_t kMaxGroupsX = 65535u;
    constexpr std::uint32_t kMaxGroupsY = 65535u;
    const std::uint32_t groups_x =
        std::min(output_dim, kMaxGroupsX);
    const std::uint64_t groups_y_64 =
        (static_cast<std::uint64_t>(output_dim) +
         groups_x - 1u) /
        groups_x;
    if (groups_y_64 > kMaxGroupsY) {
        throw std::runtime_error(
            "Q8MatVecPipeline output dimension exceeds 2D dispatch capacity");
    }

    const Push push{
        weight_byte_offset,
        input_dim,
        output_dim,
        groups_x,
    };
    vkCmdPushConstants(
        staged_command_,
        pipeline_layout_,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(push),
        &push);

    vkCmdDispatch(
        staged_command_,
        groups_x,
        static_cast<std::uint32_t>(groups_y_64),
        1);

    VkMemoryBarrier output_barrier{};
    output_barrier.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    output_barrier.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT;
    output_barrier.dstAccessMask =
        readback
            ? VK_ACCESS_TRANSFER_READ_BIT
            : VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        staged_command_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        readback
            ? VK_PIPELINE_STAGE_TRANSFER_BIT
            : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &output_barrier,
        0,
        nullptr,
        0,
        nullptr);

    if (readback) {
        VkBufferCopy output_copy{};
        output_copy.size =
            static_cast<VkDeviceSize>(required_output_bytes);
        vkCmdCopyBuffer(
            staged_command_,
            output.handle(),
            staging_output.handle(),
            1,
            &output_copy);

        VkMemoryBarrier host_barrier{};
        host_barrier.sType =
            VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        host_barrier.srcAccessMask =
            VK_ACCESS_TRANSFER_WRITE_BIT;
        host_barrier.dstAccessMask =
            VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(
            staged_command_,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1,
            &host_barrier,
            0,
            nullptr,
            0,
            nullptr);
    }

    check(
        vkEndCommandBuffer(staged_command_),
        "vkEndCommandBuffer failed for staged Q8 path");

    check(
        vkResetFences(context_.device(), 1, &fence_),
        "vkResetFences failed");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &staged_command_;

    check(
        vkQueueSubmit(
            context_.compute_queue(),
            1,
            &submit,
            fence_),
        "vkQueueSubmit failed for staged Q8 path");
    check(
        vkWaitForFences(
            context_.device(),
            1,
            &fence_,
            VK_TRUE,
            UINT64_MAX),
        "vkWaitForFences failed for staged Q8 path");

    if (readback) {
        staging_output.download(
            host_output,
            static_cast<std::size_t>(required_output_bytes));
    }
}

void Q8MatVecPipeline::run_staged_pair(
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
    const std::array<std::uint32_t, 2>& output_dims) {

    if (host_input == nullptr ||
        host_output == nullptr) {
        throw std::runtime_error(
            "Q8MatVecPipeline::run_staged_pair received null host buffer");
    }
    if (input_dim == 0 ||
        (input_dim % kQ8BlockElements) != 0) {
        throw std::runtime_error(
            "Q8MatVecPipeline pair input_dim must be non-zero and divisible by 32");
    }

    const std::uint64_t required_input_bytes =
        checked_mul(
            input_dim,
            sizeof(float),
            "Q8 pair input byte size overflow");

    if (required_input_bytes > input.size() ||
        required_input_bytes > staging_input.size() ||
        host_input_bytes < required_input_bytes) {
        throw std::runtime_error(
            "Q8 pair input buffer is too small");
    }

    const std::uint64_t blocks =
        static_cast<std::uint64_t>(input_dim) /
        kQ8BlockElements;
    const std::uint64_t row_bytes =
        checked_mul(
            blocks,
            kQ8BlockBytes,
            "Q8 pair row byte size overflow");

    std::array<VkDeviceSize, 2> output_offsets{};
    std::uint64_t total_output_bytes = 0;

    for (std::size_t i = 0;
         i < output_dims.size();
         ++i) {

        if (output_dims[i] == 0) {
            throw std::runtime_error(
                "Q8 pair output dimensions must be non-zero");
        }
        if ((weight_byte_offsets[i] & 3u) != 0u) {
            throw std::runtime_error(
                "Q8 pair weight offsets must be 4-byte aligned");
        }

        const std::uint64_t matrix_bytes =
            checked_mul(
                output_dims[i],
                row_bytes,
                "Q8 pair matrix byte size overflow");
        const std::uint64_t weight_end =
            checked_add(
                weight_byte_offsets[i],
                matrix_bytes,
                "Q8 pair weight range overflow");

        if (weight_end > weights.size()) {
            throw std::runtime_error(
                "Q8 pair weight buffer is too small");
        }

        output_offsets[i] =
            static_cast<VkDeviceSize>(
                total_output_bytes);

        total_output_bytes =
            checked_add(
                total_output_bytes,
                checked_mul(
                    output_dims[i],
                    sizeof(float),
                    "Q8 pair output byte size overflow"),
                "Q8 pair packed output size overflow");
    }

    if (total_output_bytes > output.size() ||
        total_output_bytes > staging_output.size() ||
        host_output_bytes < total_output_bytes) {
        throw std::runtime_error(
            "Q8 pair output buffer is too small");
    }

    staging_input.upload(
        host_input,
        static_cast<std::size_t>(
            required_input_bytes));

    for (std::size_t set_index = 0;
         set_index < triplet_sets_.size();
         ++set_index) {

        const VkDeviceSize output_bytes =
            static_cast<VkDeviceSize>(
                output_dims[set_index]) *
            sizeof(float);

        std::array<VkDescriptorBufferInfo, 3> infos{{
            {
                weights.handle(),
                0,
                weights.size(),
            },
            {
                input.handle(),
                0,
                input.size(),
            },
            {
                output.handle(),
                output_offsets[set_index],
                output_bytes,
            },
        }};

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t binding = 0;
             binding < writes.size();
             ++binding) {
            writes[binding].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet =
                triplet_sets_[set_index];
            writes[binding].dstBinding =
                binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo =
                &infos[binding];
        }

        vkUpdateDescriptorSets(
            context_.device(),
            static_cast<std::uint32_t>(
                writes.size()),
            writes.data(),
            0,
            nullptr);
    }

    if (triplet_command_ == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo allocate{};
        allocate.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool =
            command_pool_;
        allocate.level =
            VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;

        check(
            vkAllocateCommandBuffers(
                context_.device(),
                &allocate,
                &triplet_command_),
            "vkAllocateCommandBuffers failed for Q8 pair");
    }

    check(
        vkResetCommandBuffer(
            triplet_command_,
            0),
        "vkResetCommandBuffer failed for Q8 pair");

    VkCommandBufferBeginInfo begin{};
    begin.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    check(
        vkBeginCommandBuffer(
            triplet_command_,
            &begin),
        "vkBeginCommandBuffer failed for Q8 pair");

    VkBufferCopy input_copy{};
    input_copy.size =
        static_cast<VkDeviceSize>(
            required_input_bytes);

    vkCmdCopyBuffer(
        triplet_command_,
        staging_input.handle(),
        input.handle(),
        1,
        &input_copy);

    VkMemoryBarrier input_barrier{};
    input_barrier.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT;
    input_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        triplet_command_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &input_barrier,
        0,
        nullptr,
        0,
        nullptr);

    vkCmdBindPipeline(
        triplet_command_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_);

    constexpr std::uint32_t kMaxGroupsX =
        65535u;
    constexpr std::uint32_t kMaxGroupsY =
        65535u;

    for (std::size_t projection = 0;
         projection < output_dims.size();
         ++projection) {

        vkCmdBindDescriptorSets(
            triplet_command_,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_,
            0,
            1,
            &triplet_sets_[projection],
            0,
            nullptr);

        const std::uint32_t groups_x =
            std::min(
                output_dims[projection],
                kMaxGroupsX);
        const std::uint64_t groups_y_64 =
            (static_cast<std::uint64_t>(
                 output_dims[projection]) +
             groups_x - 1u) /
            groups_x;

        if (groups_y_64 > kMaxGroupsY) {
            throw std::runtime_error(
                "Q8 pair output dimension exceeds dispatch capacity");
        }

        const Push push{
            weight_byte_offsets[projection],
            input_dim,
            output_dims[projection],
            groups_x,
        };

        vkCmdPushConstants(
            triplet_command_,
            pipeline_layout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(push),
            &push);

        vkCmdDispatch(
            triplet_command_,
            groups_x,
            static_cast<std::uint32_t>(
                groups_y_64),
            1);
    }

    VkMemoryBarrier output_barrier{};
    output_barrier.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    output_barrier.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT;
    output_barrier.dstAccessMask =
        VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        triplet_command_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &output_barrier,
        0,
        nullptr,
        0,
        nullptr);

    VkBufferCopy output_copy{};
    output_copy.size =
        static_cast<VkDeviceSize>(
            total_output_bytes);

    vkCmdCopyBuffer(
        triplet_command_,
        output.handle(),
        staging_output.handle(),
        1,
        &output_copy);

    VkMemoryBarrier host_barrier{};
    host_barrier.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_barrier.srcAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT;
    host_barrier.dstAccessMask =
        VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        triplet_command_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &host_barrier,
        0,
        nullptr,
        0,
        nullptr);

    check(
        vkEndCommandBuffer(
            triplet_command_),
        "vkEndCommandBuffer failed for Q8 pair");

    check(
        vkResetFences(
            context_.device(),
            1,
            &fence_),
        "vkResetFences failed for Q8 pair");

    VkSubmitInfo submit{};
    submit.sType =
        VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers =
        &triplet_command_;

    check(
        vkQueueSubmit(
            context_.compute_queue(),
            1,
            &submit,
            fence_),
        "vkQueueSubmit failed for Q8 pair");

    check(
        vkWaitForFences(
            context_.device(),
            1,
            &fence_,
            VK_TRUE,
            UINT64_MAX),
        "vkWaitForFences failed for Q8 pair");

    staging_output.download(
        host_output,
        static_cast<std::size_t>(
            total_output_bytes));
}


void Q8MatVecPipeline::run_staged_triplet(
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
    const std::array<std::uint32_t, 3>& output_dims) {

    if (host_input == nullptr ||
        host_output == nullptr) {
        throw std::runtime_error(
            "Q8MatVecPipeline::run_staged_triplet received null host buffer");
    }
    if (input_dim == 0 ||
        (input_dim % kQ8BlockElements) != 0) {
        throw std::runtime_error(
            "Q8MatVecPipeline triplet input_dim must be non-zero and divisible by 32");
    }

    const std::uint64_t required_input_bytes =
        checked_mul(
            input_dim,
            sizeof(float),
            "Q8 triplet input byte size overflow");

    if (required_input_bytes > input.size() ||
        required_input_bytes > staging_input.size() ||
        host_input_bytes < required_input_bytes) {
        throw std::runtime_error(
            "Q8 triplet input buffer is too small");
    }

    const std::uint64_t blocks =
        static_cast<std::uint64_t>(input_dim) /
        kQ8BlockElements;
    const std::uint64_t row_bytes =
        checked_mul(
            blocks,
            kQ8BlockBytes,
            "Q8 triplet row byte size overflow");

    std::array<VkDeviceSize, 3> output_offsets{};
    std::uint64_t total_output_bytes = 0;

    for (std::size_t i = 0;
         i < output_dims.size();
         ++i) {

        if (output_dims[i] == 0) {
            throw std::runtime_error(
                "Q8 triplet output dimensions must be non-zero");
        }
        if ((weight_byte_offsets[i] & 3u) != 0u) {
            throw std::runtime_error(
                "Q8 triplet weight offsets must be 4-byte aligned");
        }

        const std::uint64_t matrix_bytes =
            checked_mul(
                output_dims[i],
                row_bytes,
                "Q8 triplet matrix byte size overflow");
        const std::uint64_t weight_end =
            checked_add(
                weight_byte_offsets[i],
                matrix_bytes,
                "Q8 triplet weight range overflow");

        if (weight_end > weights.size()) {
            throw std::runtime_error(
                "Q8 triplet weight buffer is too small");
        }

        output_offsets[i] =
            static_cast<VkDeviceSize>(
                total_output_bytes);

        total_output_bytes =
            checked_add(
                total_output_bytes,
                checked_mul(
                    output_dims[i],
                    sizeof(float),
                    "Q8 triplet output byte size overflow"),
                "Q8 triplet packed output size overflow");
    }

    if (total_output_bytes > output.size() ||
        total_output_bytes > staging_output.size() ||
        host_output_bytes < total_output_bytes) {
        throw std::runtime_error(
            "Q8 triplet output buffer is too small");
    }

    staging_input.upload(
        host_input,
        static_cast<std::size_t>(
            required_input_bytes));

    for (std::size_t set_index = 0;
         set_index < triplet_sets_.size();
         ++set_index) {

        const VkDeviceSize output_bytes =
            static_cast<VkDeviceSize>(
                output_dims[set_index]) *
            sizeof(float);

        std::array<VkDescriptorBufferInfo, 3> infos{{
            {
                weights.handle(),
                0,
                weights.size(),
            },
            {
                input.handle(),
                0,
                input.size(),
            },
            {
                output.handle(),
                output_offsets[set_index],
                output_bytes,
            },
        }};

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t binding = 0;
             binding < writes.size();
             ++binding) {
            writes[binding].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet =
                triplet_sets_[set_index];
            writes[binding].dstBinding =
                binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo =
                &infos[binding];
        }

        vkUpdateDescriptorSets(
            context_.device(),
            static_cast<std::uint32_t>(
                writes.size()),
            writes.data(),
            0,
            nullptr);
    }

    if (triplet_command_ == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo allocate{};
        allocate.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool =
            command_pool_;
        allocate.level =
            VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;

        check(
            vkAllocateCommandBuffers(
                context_.device(),
                &allocate,
                &triplet_command_),
            "vkAllocateCommandBuffers failed for Q8 triplet");
    }

    check(
        vkResetCommandBuffer(
            triplet_command_,
            0),
        "vkResetCommandBuffer failed for Q8 triplet");

    VkCommandBufferBeginInfo begin{};
    begin.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    check(
        vkBeginCommandBuffer(
            triplet_command_,
            &begin),
        "vkBeginCommandBuffer failed for Q8 triplet");

    VkBufferCopy input_copy{};
    input_copy.size =
        static_cast<VkDeviceSize>(
            required_input_bytes);

    vkCmdCopyBuffer(
        triplet_command_,
        staging_input.handle(),
        input.handle(),
        1,
        &input_copy);

    VkMemoryBarrier input_barrier{};
    input_barrier.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT;
    input_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        triplet_command_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &input_barrier,
        0,
        nullptr,
        0,
        nullptr);

    vkCmdBindPipeline(
        triplet_command_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_);

    constexpr std::uint32_t kMaxGroupsX =
        65535u;
    constexpr std::uint32_t kMaxGroupsY =
        65535u;

    for (std::size_t projection = 0;
         projection < output_dims.size();
         ++projection) {

        vkCmdBindDescriptorSets(
            triplet_command_,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_,
            0,
            1,
            &triplet_sets_[projection],
            0,
            nullptr);

        const std::uint32_t groups_x =
            std::min(
                output_dims[projection],
                kMaxGroupsX);
        const std::uint64_t groups_y_64 =
            (static_cast<std::uint64_t>(
                 output_dims[projection]) +
             groups_x - 1u) /
            groups_x;

        if (groups_y_64 > kMaxGroupsY) {
            throw std::runtime_error(
                "Q8 triplet output dimension exceeds dispatch capacity");
        }

        const Push push{
            weight_byte_offsets[projection],
            input_dim,
            output_dims[projection],
            groups_x,
        };

        vkCmdPushConstants(
            triplet_command_,
            pipeline_layout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(push),
            &push);

        vkCmdDispatch(
            triplet_command_,
            groups_x,
            static_cast<std::uint32_t>(
                groups_y_64),
            1);
    }

    VkMemoryBarrier output_barrier{};
    output_barrier.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    output_barrier.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT;
    output_barrier.dstAccessMask =
        VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        triplet_command_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &output_barrier,
        0,
        nullptr,
        0,
        nullptr);

    VkBufferCopy output_copy{};
    output_copy.size =
        static_cast<VkDeviceSize>(
            total_output_bytes);

    vkCmdCopyBuffer(
        triplet_command_,
        output.handle(),
        staging_output.handle(),
        1,
        &output_copy);

    VkMemoryBarrier host_barrier{};
    host_barrier.sType =
        VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_barrier.srcAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT;
    host_barrier.dstAccessMask =
        VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        triplet_command_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &host_barrier,
        0,
        nullptr,
        0,
        nullptr);

    check(
        vkEndCommandBuffer(
            triplet_command_),
        "vkEndCommandBuffer failed for Q8 triplet");

    check(
        vkResetFences(
            context_.device(),
            1,
            &fence_),
        "vkResetFences failed for Q8 triplet");

    VkSubmitInfo submit{};
    submit.sType =
        VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers =
        &triplet_command_;

    check(
        vkQueueSubmit(
            context_.compute_queue(),
            1,
            &submit,
            fence_),
        "vkQueueSubmit failed for Q8 triplet");

    check(
        vkWaitForFences(
            context_.device(),
            1,
            &fence_,
            VK_TRUE,
            UINT64_MAX),
        "vkWaitForFences failed for Q8 triplet");

    staging_output.download(
        host_output,
        static_cast<std::size_t>(
            total_output_bytes));
}

} // namespace vortexrt
