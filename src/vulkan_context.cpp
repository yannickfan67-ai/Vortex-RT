#include "vortexrt/vulkan_context.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vortexrt {
namespace {

constexpr VkDeviceSize kDefaultMemoryBlockSize = 128ull * 1024ull * 1024ull;
constexpr VkDeviceSize kDefaultStagingSize = 64ull * 1024ull * 1024ull;

void vk_check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(what);
    }
}

VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1) {
        return value;
    }
    const VkDeviceSize remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    const VkDeviceSize delta = alignment - remainder;
    if (value > std::numeric_limits<VkDeviceSize>::max() - delta) {
        throw std::runtime_error("Vulkan allocation alignment overflow");
    }
    return value + delta;
}

std::string lowercase(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool parse_index(const std::string& text, std::uint32_t& value) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

std::uint32_t choose_compute_queue(
    VkPhysicalDevice device,
    std::uint32_t& timestamp_valid_bits) {

    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());

    std::uint32_t fallback = UINT32_MAX;

    for (std::uint32_t i = 0; i < count; ++i) {
        const bool compute = (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool graphics = (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;

        if (!compute) {
            continue;
        }

        if (fallback == UINT32_MAX) {
            fallback = i;
        }

        if (!graphics) {
            timestamp_valid_bits = props[i].timestampValidBits;
            return i;
        }
    }

    if (fallback != UINT32_MAX) {
        timestamp_valid_bits = props[fallback].timestampValidBits;
    }

    return fallback;
}

int score_device(const VkPhysicalDeviceProperties& props) {
    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 10000;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += 5000;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
        score += 2500;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        score += 1000;
    }

    score += static_cast<int>(props.limits.maxComputeSharedMemorySize / 1024);
    return score;
}

} // namespace

VulkanContext::VulkanContext(std::string device_selector)
    : device_selector_(std::move(device_selector)) {

    if (device_selector_.empty()) {
        if (const char* env = std::getenv("VORTEXRT_DEVICE"); env != nullptr) {
            device_selector_ = env;
        }
    }

    try {
        create_instance();
        select_physical_device();
        create_device();
        create_transfer_resources();
    } catch (...) {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            destroy_transfer_resources();
            destroy_memory_blocks();
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
        throw;
    }
}

VulkanContext::~VulkanContext() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        destroy_transfer_resources();
        destroy_memory_blocks();
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void VulkanContext::create_instance() {
    std::uint32_t loader_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) {
        vkEnumerateInstanceVersion(&loader_version);
    }

    if (loader_version < VK_API_VERSION_1_3) {
        throw std::runtime_error("Vortex-RT requires a Vulkan 1.3 loader");
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Vortex-RT";
    app.applicationVersion = VK_MAKE_VERSION(0, 2, 0);
    app.pEngineName = "Vortex-RT";
    app.engineVersion = VK_MAKE_VERSION(0, 2, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    vk_check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance failed");
}

void VulkanContext::select_physical_device() {
    std::uint32_t count = 0;
    vk_check(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
             "vkEnumeratePhysicalDevices failed");

    if (count == 0) {
        throw std::runtime_error("No Vulkan physical device found");
    }

    std::vector<VkPhysicalDevice> devices(count);
    vk_check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
             "vkEnumeratePhysicalDevices failed");

    struct Candidate {
        VkPhysicalDevice device = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties props{};
        std::uint32_t queue_family = UINT32_MAX;
        std::uint32_t timestamp_bits = 0;
        std::uint32_t index = 0;
        int score = 0;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(devices.size());

    for (std::uint32_t i = 0; i < devices.size(); ++i) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[i], &props);

        if (props.apiVersion < VK_API_VERSION_1_3) {
            continue;
        }

        std::uint32_t timestamp_bits = 0;
        const std::uint32_t queue_family =
            choose_compute_queue(devices[i], timestamp_bits);
        if (queue_family == UINT32_MAX) {
            continue;
        }

        Candidate candidate{};
        candidate.device = devices[i];
        candidate.props = props;
        candidate.queue_family = queue_family;
        candidate.timestamp_bits = timestamp_bits;
        candidate.index = i;
        candidate.score = score_device(props);
        candidates.push_back(candidate);
    }

    if (candidates.empty()) {
        throw std::runtime_error("No Vulkan 1.3 compute-capable device found");
    }

    const Candidate* selected = nullptr;

    if (!device_selector_.empty()) {
        std::uint32_t requested_index = 0;
        if (parse_index(device_selector_, requested_index)) {
            for (const auto& candidate : candidates) {
                if (candidate.index == requested_index) {
                    selected = &candidate;
                    break;
                }
            }
        } else {
            const std::string needle = lowercase(device_selector_);
            for (const auto& candidate : candidates) {
                if (lowercase(candidate.props.deviceName).find(needle) != std::string::npos &&
                    (selected == nullptr || candidate.score > selected->score)) {
                    selected = &candidate;
                }
            }
        }

        if (selected == nullptr) {
            throw std::runtime_error(
                "Requested Vulkan device not found: " + device_selector_);
        }
    } else {
        selected = &*std::max_element(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.score < b.score;
            });
    }

    physical_device_ = selected->device;
    compute_queue_family_ = selected->queue_family;
    caps_.timestamp_valid_bits = selected->timestamp_bits;
    caps_.device_index = selected->index;

    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical_device_, &props2);

    supported11_ = {};
    supported11_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    supported12_ = {};
    supported12_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    supported11_.pNext = &supported12_;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &supported11_;
    vkGetPhysicalDeviceFeatures2(physical_device_, &features2);

    caps_.name = props2.properties.deviceName;
    caps_.vendor_id = props2.properties.vendorID;
    caps_.device_id = props2.properties.deviceID;
    caps_.device_type = props2.properties.deviceType;
    caps_.api_version = props2.properties.apiVersion;
    caps_.subgroup_size = subgroup.subgroupSize;
    caps_.subgroup_ops = subgroup.supportedOperations;
    caps_.fp16 = supported12_.shaderFloat16 == VK_TRUE;
    caps_.int8 = supported12_.shaderInt8 == VK_TRUE;
    caps_.storage8 = supported12_.storageBuffer8BitAccess == VK_TRUE;
    caps_.storage16 = supported11_.storageBuffer16BitAccess == VK_TRUE;
    caps_.timestamp_period_ns = props2.properties.limits.timestampPeriod;
    caps_.non_coherent_atom_size =
        std::max<VkDeviceSize>(1, props2.properties.limits.nonCoherentAtomSize);

    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);
    for (std::uint32_t i = 0; i < memory_properties_.memoryHeapCount; ++i) {
        if ((memory_properties_.memoryHeaps[i].flags &
             VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            caps_.device_local_bytes += memory_properties_.memoryHeaps[i].size;
        }
    }
}

void VulkanContext::create_device() {
    const float priority = 1.0f;

    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = compute_queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan11Features enabled11{};
    enabled11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    enabled11.storageBuffer16BitAccess = supported11_.storageBuffer16BitAccess;

    VkPhysicalDeviceVulkan12Features enabled12{};
    enabled12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enabled12.storageBuffer8BitAccess = supported12_.storageBuffer8BitAccess;
    enabled12.shaderFloat16 = supported12_.shaderFloat16;
    enabled12.shaderInt8 = supported12_.shaderInt8;

    enabled11.pNext = &enabled12;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &enabled11;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    vk_check(vkCreateDevice(physical_device_, &dci, nullptr, &device_),
             "vkCreateDevice failed");

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
}

void VulkanContext::create_transfer_resources() {
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags =
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = compute_queue_family_;

    vk_check(
        vkCreateCommandPool(device_, &pool_ci, nullptr, &transfer_command_pool_),
        "vkCreateCommandPool failed for transfer path");

    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = transfer_command_pool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;

    vk_check(
        vkAllocateCommandBuffers(device_, &alloc, &transfer_command_buffer_),
        "vkAllocateCommandBuffers failed for transfer path");

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_check(
        vkCreateFence(device_, &fci, nullptr, &transfer_fence_),
        "vkCreateFence failed for transfer path");
}

void VulkanContext::destroy_transfer_resources() noexcept {
    if (staging_mapped_ != nullptr && staging_memory_ != VK_NULL_HANDLE) {
        vkUnmapMemory(device_, staging_memory_);
        staging_mapped_ = nullptr;
    }
    if (staging_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, staging_buffer_, nullptr);
        staging_buffer_ = VK_NULL_HANDLE;
    }
    if (staging_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, staging_memory_, nullptr);
        staging_memory_ = VK_NULL_HANDLE;
    }
    staging_capacity_ = 0;

    if (transfer_fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, transfer_fence_, nullptr);
        transfer_fence_ = VK_NULL_HANDLE;
    }
    if (transfer_command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, transfer_command_pool_, nullptr);
        transfer_command_pool_ = VK_NULL_HANDLE;
        transfer_command_buffer_ = VK_NULL_HANDLE;
    }
}

std::uint32_t VulkanContext::find_memory_type(
    std::uint32_t type_bits,
    VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred) const {

    auto find = [&](VkMemoryPropertyFlags wanted) -> std::uint32_t {
        for (std::uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
            const bool allowed = (type_bits & (1u << i)) != 0;
            const auto flags = memory_properties_.memoryTypes[i].propertyFlags;
            if (allowed && (flags & wanted) == wanted) {
                return i;
            }
        }
        return UINT32_MAX;
    };

    if (preferred != 0) {
        const auto preferred_type = find(required | preferred);
        if (preferred_type != UINT32_MAX) {
            return preferred_type;
        }
    }

    const auto required_type = find(required);
    if (required_type != UINT32_MAX) {
        return required_type;
    }

    throw std::runtime_error("No compatible Vulkan memory type found");
}

BufferMemoryAllocation VulkanContext::allocate_buffer_memory(
    const VkMemoryRequirements& requirements,
    VkMemoryPropertyFlags properties) {

    const std::uint32_t memory_type =
        find_memory_type(requirements.memoryTypeBits, properties);

    if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = requirements.size;
        mai.memoryTypeIndex = memory_type;

        BufferMemoryAllocation allocation{};
        vk_check(
            vkAllocateMemory(device_, &mai, nullptr, &allocation.memory),
            "vkAllocateMemory failed");
        allocation.size = requirements.size;
        return allocation;
    }

    std::lock_guard lock(memory_mutex_);

    auto allocate_from_block =
        [&](MemoryBlock& block) -> BufferMemoryAllocation {
            for (std::size_t i = 0; i < block.free_ranges.size(); ++i) {
                const FreeRange range = block.free_ranges[i];
                const VkDeviceSize offset =
                    align_up(range.offset, requirements.alignment);

                if (offset < range.offset ||
                    offset > range.offset + range.size ||
                    requirements.size > range.offset + range.size - offset) {
                    continue;
                }

                const VkDeviceSize before = offset - range.offset;
                const VkDeviceSize after_offset = offset + requirements.size;
                const VkDeviceSize after =
                    range.offset + range.size - after_offset;

                block.free_ranges.erase(block.free_ranges.begin() + i);
                if (after != 0) {
                    block.free_ranges.insert(
                        block.free_ranges.begin() + i,
                        FreeRange{after_offset, after});
                }
                if (before != 0) {
                    block.free_ranges.insert(
                        block.free_ranges.begin() + i,
                        FreeRange{range.offset, before});
                }

                ++block.live_allocations;

                BufferMemoryAllocation allocation{};
                allocation.memory = block.memory;
                allocation.offset = offset;
                allocation.size = requirements.size;
                allocation.block_id = block.id;
                allocation.pooled = true;
                return allocation;
            }
            return {};
        };

    for (auto& block : memory_blocks_) {
        if (block.memory_type == memory_type) {
            if (auto allocation = allocate_from_block(block);
                allocation.memory != VK_NULL_HANDLE) {
                return allocation;
            }
        }
    }

    VkDeviceSize block_size =
        std::max(kDefaultMemoryBlockSize,
                 align_up(requirements.size, requirements.alignment));

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = block_size;
    mai.memoryTypeIndex = memory_type;

    VkResult result = vkAllocateMemory(device_, &mai, nullptr, &memory);
    if (result != VK_SUCCESS && block_size != requirements.size) {
        block_size = requirements.size;
        mai.allocationSize = block_size;
        result = vkAllocateMemory(device_, &mai, nullptr, &memory);
    }
    vk_check(result, "vkAllocateMemory failed for pooled device memory");

    MemoryBlock block{};
    block.id = next_memory_block_id_++;
    block.memory = memory;
    block.size = block_size;
    block.memory_type = memory_type;
    block.free_ranges.push_back(FreeRange{0, block_size});
    memory_blocks_.push_back(std::move(block));

    auto allocation = allocate_from_block(memory_blocks_.back());
    if (allocation.memory == VK_NULL_HANDLE) {
        throw std::runtime_error("Internal device-memory suballocation failure");
    }
    return allocation;
}

void VulkanContext::free_buffer_memory(
    BufferMemoryAllocation& allocation) noexcept {

    if (allocation.memory == VK_NULL_HANDLE) {
        return;
    }

    if (!allocation.pooled) {
        vkFreeMemory(device_, allocation.memory, nullptr);
        allocation = {};
        return;
    }

    std::lock_guard lock(memory_mutex_);

    for (auto it = memory_blocks_.begin(); it != memory_blocks_.end(); ++it) {
        if (it->id != allocation.block_id) {
            continue;
        }

        it->free_ranges.push_back(FreeRange{allocation.offset, allocation.size});
        std::sort(
            it->free_ranges.begin(),
            it->free_ranges.end(),
            [](const FreeRange& a, const FreeRange& b) {
                return a.offset < b.offset;
            });

        std::vector<FreeRange> merged;
        merged.reserve(it->free_ranges.size());
        for (const auto& range : it->free_ranges) {
            if (!merged.empty() &&
                merged.back().offset + merged.back().size == range.offset) {
                merged.back().size += range.size;
            } else {
                merged.push_back(range);
            }
        }
        it->free_ranges = std::move(merged);

        if (it->live_allocations != 0) {
            --it->live_allocations;
        }

        if (it->live_allocations == 0) {
            vkFreeMemory(device_, it->memory, nullptr);
            memory_blocks_.erase(it);
        }

        allocation = {};
        return;
    }

    allocation = {};
}

void VulkanContext::destroy_memory_blocks() noexcept {
    std::lock_guard lock(memory_mutex_);
    for (auto& block : memory_blocks_) {
        if (block.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, block.memory, nullptr);
        }
    }
    memory_blocks_.clear();
}

void VulkanContext::copy_buffer(
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize bytes,
    VkDeviceSize src_offset,
    VkDeviceSize dst_offset) {

    if (bytes == 0) {
        return;
    }

    std::lock_guard lock(transfer_mutex_);
    copy_buffer_unlocked(src, dst, bytes, src_offset, dst_offset);
}

void VulkanContext::copy_buffer_unlocked(
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize bytes,
    VkDeviceSize src_offset,
    VkDeviceSize dst_offset) {

    vk_check(
        vkResetFences(device_, 1, &transfer_fence_),
        "vkResetFences failed for transfer path");
    vk_check(
        vkResetCommandPool(device_, transfer_command_pool_, 0),
        "vkResetCommandPool failed for transfer path");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vk_check(
        vkBeginCommandBuffer(transfer_command_buffer_, &begin),
        "vkBeginCommandBuffer failed for transfer path");

    VkBufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = bytes;
    vkCmdCopyBuffer(transfer_command_buffer_, src, dst, 1, &region);

    vk_check(
        vkEndCommandBuffer(transfer_command_buffer_),
        "vkEndCommandBuffer failed for transfer path");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &transfer_command_buffer_;

    vk_check(
        vkQueueSubmit(compute_queue_, 1, &submit, transfer_fence_),
        "vkQueueSubmit failed for transfer path");
    vk_check(
        vkWaitForFences(device_, 1, &transfer_fence_, VK_TRUE, UINT64_MAX),
        "vkWaitForFences failed for transfer path");
}

void VulkanContext::ensure_staging_capacity(VkDeviceSize bytes) {
    if (bytes <= staging_capacity_) {
        return;
    }

    VkDeviceSize capacity = std::max(kDefaultStagingSize, staging_capacity_);
    while (capacity < bytes) {
        if (capacity > std::numeric_limits<VkDeviceSize>::max() / 2) {
            capacity = bytes;
            break;
        }
        capacity *= 2;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;

    try {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = capacity;
        bci.usage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vk_check(
            vkCreateBuffer(device_, &bci, nullptr, &buffer),
            "vkCreateBuffer failed for staging arena");

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, buffer, &req);

        const auto memory_type = find_memory_type(
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type;

        vk_check(
            vkAllocateMemory(device_, &mai, nullptr, &memory),
            "vkAllocateMemory failed for staging arena");
        vk_check(
            vkBindBufferMemory(device_, buffer, memory, 0),
            "vkBindBufferMemory failed for staging arena");
        vk_check(
            vkMapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &mapped),
            "vkMapMemory failed for staging arena");

        if (staging_mapped_ != nullptr && staging_memory_ != VK_NULL_HANDLE) {
            vkUnmapMemory(device_, staging_memory_);
        }
        if (staging_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, staging_buffer_, nullptr);
        }
        if (staging_memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, staging_memory_, nullptr);
        }

        staging_buffer_ = buffer;
        staging_memory_ = memory;
        staging_mapped_ = mapped;
        staging_capacity_ = capacity;
        staging_memory_properties_ =
            memory_properties_.memoryTypes[memory_type].propertyFlags;
    } catch (...) {
        if (mapped != nullptr && memory != VK_NULL_HANDLE) {
            vkUnmapMemory(device_, memory);
        }
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer, nullptr);
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, memory, nullptr);
        }
        throw;
    }
}

void VulkanContext::stage_upload(
    VkBuffer dst,
    const void* data,
    std::size_t bytes,
    VkDeviceSize dst_offset) {

    if (bytes == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::runtime_error("stage_upload received null data");
    }

    std::lock_guard lock(transfer_mutex_);
    ensure_staging_capacity(static_cast<VkDeviceSize>(bytes));

    std::memcpy(staging_mapped_, data, bytes);

    if ((staging_memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = staging_memory_;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vk_check(
            vkFlushMappedMemoryRanges(device_, 1, &range),
            "vkFlushMappedMemoryRanges failed for staging upload");
    }

    copy_buffer_unlocked(
        staging_buffer_,
        dst,
        static_cast<VkDeviceSize>(bytes),
        0,
        dst_offset);
}

void VulkanContext::stage_download(
    VkBuffer src,
    void* data,
    std::size_t bytes,
    VkDeviceSize src_offset) {

    if (bytes == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::runtime_error("stage_download received null data");
    }

    std::lock_guard lock(transfer_mutex_);
    ensure_staging_capacity(static_cast<VkDeviceSize>(bytes));

    copy_buffer_unlocked(
        src,
        staging_buffer_,
        static_cast<VkDeviceSize>(bytes),
        src_offset,
        0);

    if ((staging_memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = staging_memory_;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vk_check(
            vkInvalidateMappedMemoryRanges(device_, 1, &range),
            "vkInvalidateMappedMemoryRanges failed for staging download");
    }

    std::memcpy(data, staging_mapped_, bytes);
}

} // namespace vortexrt
