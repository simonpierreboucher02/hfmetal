#pragma once

#include <string>
#include <string_view>
#include <variant>
#include <stdexcept>

namespace hfm {

enum class ErrorCode : int {
    Ok = 0,
    InvalidArgument,
    OutOfRange,
    DimensionMismatch,
    SingularMatrix,
    NotConverged,
    MetalUnavailable,
    MetalKernelError,
    AllocationFailed,
    NotImplemented,
    IOError,
    InternalError
};

class Status {
public:
    Status() : code_(ErrorCode::Ok) {}
    Status(ErrorCode code, std::string msg) : code_(code), message_(std::move(msg)) {}

    static Status ok() { return Status(); }
    static Status error(ErrorCode code, std::string msg) { return Status(code, std::move(msg)); }

    bool is_ok() const { return code_ == ErrorCode::Ok; }
    explicit operator bool() const { return is_ok(); }

    ErrorCode code() const { return code_; }
    const std::string& message() const { return message_; }

    void throw_if_error() const {
        if (!is_ok()) {
            throw std::runtime_error(message_);
        }
    }

private:
    ErrorCode code_;
    std::string message_;
};

template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Status status) : data_(std::move(status)) {}

    bool is_ok() const { return std::holds_alternative<T>(data_); }
    explicit operator bool() const { return is_ok(); }

    const T& value() const& {
        if (!is_ok()) status().throw_if_error();
        return std::get<T>(data_);
    }

    T&& value() && {
        if (!is_ok()) status().throw_if_error();
        return std::get<T>(std::move(data_));
    }

    const Status& status() const {
        return std::get<Status>(data_);
    }

private:
    std::variant<T, Status> data_;
};

} // namespace hfm
