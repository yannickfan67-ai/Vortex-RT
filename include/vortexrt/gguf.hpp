#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace vortexrt {

enum class GgufValueType : std::uint32_t {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
};

using GgufScalar = std::variant<
    std::monostate,
    std::uint64_t,
    std::int64_t,
    double,
    bool,
    std::string>;

struct GgufValue {
    GgufValueType type = GgufValueType::UInt8;
    GgufScalar scalar{};
    GgufValueType array_element_type = GgufValueType::UInt8;
    std::vector<GgufValue> array{};

    [[nodiscard]] bool is_array() const noexcept {
        return type == GgufValueType::Array;
    }

    [[nodiscard]] std::optional<std::string> as_string() const;
    [[nodiscard]] std::optional<std::uint64_t> as_u64() const;
    [[nodiscard]] std::optional<std::int64_t> as_i64() const;
    [[nodiscard]] std::optional<double> as_f64() const;
    [[nodiscard]] std::optional<bool> as_bool() const;
};

struct GgufMetadata {
    std::string key;
    GgufValue value;
};

struct GgufTensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
};

class GgufFile {
public:
    explicit GgufFile(const std::filesystem::path& path);

    [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
    [[nodiscard]] std::uint64_t file_size() const noexcept { return file_size_; }
    [[nodiscard]] std::uint64_t data_offset() const noexcept { return data_offset_; }
    [[nodiscard]] std::uint32_t alignment() const noexcept { return alignment_; }

    [[nodiscard]] const std::vector<GgufMetadata>& metadata() const noexcept {
        return metadata_;
    }

    [[nodiscard]] const std::vector<GgufTensorInfo>& tensors() const noexcept {
        return tensors_;
    }

    [[nodiscard]] const GgufValue* find_metadata(const std::string& key) const noexcept;
    [[nodiscard]] const GgufTensorInfo* find_tensor(const std::string& name) const noexcept;

    [[nodiscard]] std::optional<std::string> metadata_string(const std::string& key) const;
    [[nodiscard]] std::optional<std::uint64_t> metadata_u64(const std::string& key) const;

    [[nodiscard]] std::optional<std::uint64_t> tensor_byte_size(
        const GgufTensorInfo& tensor) const;

    [[nodiscard]] std::vector<std::byte> read_tensor_bytes(
        const GgufTensorInfo& tensor) const;

    [[nodiscard]] std::vector<float> read_f32_tensor(
        const GgufTensorInfo& tensor) const;

    [[nodiscard]] std::vector<float> read_q8_0_row(
        const GgufTensorInfo& tensor,
        std::uint64_t row) const;

    void validate() const;

private:
    void read_bytes_at(
        std::uint64_t absolute_offset,
        void* destination,
        std::size_t bytes) const;

    std::filesystem::path path_;
    std::uint32_t version_ = 0;
    std::uint64_t file_size_ = 0;
    std::uint64_t data_offset_ = 0;
    std::uint32_t alignment_ = 32;

    std::vector<GgufMetadata> metadata_;
    std::vector<GgufTensorInfo> tensors_;
    std::unordered_map<std::string, std::size_t> metadata_index_;
    std::unordered_map<std::string, std::size_t> tensor_index_;
    mutable std::mutex payload_stream_mutex_;
    mutable std::ifstream payload_stream_;
};

[[nodiscard]] const char* gguf_value_type_name(GgufValueType type) noexcept;
[[nodiscard]] const char* ggml_type_name(std::uint32_t type) noexcept;

} // namespace vortexrt
