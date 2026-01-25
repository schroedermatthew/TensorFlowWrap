/**
 * @file scope_guard.hpp
 * @brief RAII scope-exit cleanup utility for TensorFlowWrap
 * 
 * @details This is a standalone adaptation of fat_p::ScopeGuard, providing
 * automatic resource cleanup on scope exit with three execution modes:
 * 
 * - ScopeGuard: Always executes cleanup (like Go's defer)
 * - ScopeGuardOnFail: Only executes on exception (for rollback)
 * - ScopeGuardOnSuccess: Only executes on normal exit (for commit)
 * 
 * Key features:
 * - [[nodiscard]] prevents accidentally discarding the guard
 * - dismiss() allows canceling cleanup (ownership transfer)
 * - Move-only semantics ensure single ownership
 * - Zero external dependencies
 * 
 * IMPORTANT - LAMBDA CONTROL FLOW WARNING:
 * The macros (TF_SCOPE_EXIT, TF_SCOPE_FAIL, TF_SCOPE_SUCCESS) create lambdas.
 * - 'return' inside the block returns from the LAMBDA, not the enclosing function!
 * - 'break' and 'continue' are NOT valid inside the block.
 * 
 * Example usage with TensorFlow C API:
 * @code
 * TF_Tensor* tensor = TF_AllocateTensor(...);
 * auto guard = tf_wrap::makeScopeGuard([&]() noexcept { 
 *     TF_DeleteTensor(tensor); 
 * });
 * // ... operations that might throw ...
 * guard.dismiss(); // Only if transferring ownership
 * @endcode
 * 
 * Based on fat_p::ScopeGuard by cpp_utilities (2025)
 * Adapted for TensorFlowWrap - zero dependencies version
 * 
 * @copyright MIT License
 */

#pragma once

#include <exception>    // std::uncaught_exceptions
#include <type_traits>  // std::decay_t, std::is_nothrow_*
#include <utility>      // std::forward, std::move

namespace tf_wrap {

// =============================================================================
// Exception Handling Policies
// =============================================================================

/**
 * @brief Policy that requires the cleanup action to be noexcept.
 * 
 * Use when maximum safety is required and actions are guaranteed not to throw.
 * A static_assert will fail at compile time if the action is not noexcept.
 */
struct ScopeGuardNothrowPolicy {};

/**
 * @brief Policy that catches any exception and calls std::terminate().
 * 
 * This is the default policy. Use when cleanup failure should be fatal.
 */
struct ScopeGuardTerminatePolicy {};

/**
 * @brief Policy that catches any exception and suppresses it.
 * 
 * Use when cleanup errors should be non-fatal and the program should continue.
 * Errors are silently swallowed (no logging to avoid dependencies).
 */
struct ScopeGuardSwallowPolicy {};

// =============================================================================
// Policy Executor
// =============================================================================

namespace detail {

template <typename F, typename Policy>
struct ScopeGuardPolicyExecutor;

template <typename F>
struct ScopeGuardPolicyExecutor<F, ScopeGuardNothrowPolicy> {
    static_assert(std::is_nothrow_invocable_v<F>,
                  "ScopeGuardNothrowPolicy requires action to be noexcept");
    
    static void execute(F& action) noexcept {
        action();
    }
};

template <typename F>
struct ScopeGuardPolicyExecutor<F, ScopeGuardTerminatePolicy> {
    static void execute(F& action) noexcept {
        try {
            action();
        } catch (...) {
            std::terminate();
        }
    }
};

template <typename F>
struct ScopeGuardPolicyExecutor<F, ScopeGuardSwallowPolicy> {
    static void execute(F& action) noexcept {
        try {
            action();
        } catch (...) {
            // Swallow exception silently
        }
    }
};

} // namespace detail

// =============================================================================
// ScopeGuard - Always Executes
// =============================================================================

/**
 * @brief RAII utility that executes cleanup on scope exit.
 * 
 * @tparam F The type of the cleanup function object (e.g., lambda)
 * @tparam Policy The exception handling policy (default: terminate on throw)
 * 
 * The cleanup action runs when the guard is destroyed, unless dismiss() was called.
 */
template <typename F, typename Policy = ScopeGuardTerminatePolicy>
class [[nodiscard]] ScopeGuard {
public:
    /**
     * @brief Constructs the ScopeGuard with the cleanup action.
     * @param action The function object to execute on scope exit.
     */
    explicit ScopeGuard(F&& action) noexcept(std::is_nothrow_move_constructible_v<F>)
        : m_action(std::forward<F>(action))
        , m_active(true)
    {}

    /**
     * @brief Move constructor. Transfers ownership and disables source.
     */
    ScopeGuard(ScopeGuard&& other) noexcept(std::is_nothrow_move_constructible_v<F>)
        : m_action(std::move(other.m_action))
        , m_active(other.m_active)
    {
        other.m_active = false;
    }

    /**
     * @brief Move assignment. Executes current action if active before transfer.
     */
    ScopeGuard& operator=(ScopeGuard&& other) noexcept(std::is_nothrow_move_assignable_v<F>) {
        if (this != &other) {
            if (m_active) {
                detail::ScopeGuardPolicyExecutor<F, Policy>::execute(m_action);
            }
            m_action = std::move(other.m_action);
            m_active = other.m_active;
            other.m_active = false;
        }
        return *this;
    }

    // Non-copyable
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    /**
     * @brief Destructor. Executes cleanup if not dismissed.
     */
    ~ScopeGuard() noexcept {
        if (m_active) {
            detail::ScopeGuardPolicyExecutor<F, Policy>::execute(m_action);
        }
    }

    /**
     * @brief Disables the cleanup action.
     * 
     * Call when the resource is successfully transferred elsewhere
     * and cleanup is no longer needed.
     */
    void dismiss() noexcept { m_active = false; }

    /**
     * @brief Conditionally dismiss based on a condition.
     * @param condition If true, dismiss the guard.
     */
    void dismiss_if(bool condition) noexcept { 
        if (condition) m_active = false; 
    }

    /**
     * @brief Check if the guard is still active.
     * @return true if cleanup will run, false if dismissed.
     */
    [[nodiscard]] bool is_active() const noexcept { return m_active; }

private:
    F m_action;
    bool m_active;
};

// =============================================================================
// ScopeGuardOnFail - Executes Only on Exception
// =============================================================================

/**
 * @brief Scope guard that executes only when leaving scope due to an exception.
 * 
 * Uses std::uncaught_exceptions() (C++17) to detect if stack unwinding is in progress.
 * Perfect for implementing rollback semantics in transactions.
 * 
 * @tparam F The type of the cleanup function object
 * 
 * Example:
 * @code
 * void transfer(Account& from, Account& to, int amount) {
 *     from.withdraw(amount);
 *     auto rollback = makeScopeGuardOnFail([&]() noexcept {
 *         from.deposit(amount);  // Undo withdrawal on exception
 *     });
 *     to.deposit(amount);  // If this throws, withdrawal is rolled back
 * }
 * @endcode
 */
template <typename F>
class [[nodiscard]] ScopeGuardOnFail {
public:
    explicit ScopeGuardOnFail(F&& action) noexcept(std::is_nothrow_move_constructible_v<F>)
        : m_action(std::forward<F>(action))
        , m_uncaught_count(std::uncaught_exceptions())
        , m_active(true)
    {}

    ScopeGuardOnFail(ScopeGuardOnFail&& other) noexcept(std::is_nothrow_move_constructible_v<F>)
        : m_action(std::move(other.m_action))
        , m_uncaught_count(other.m_uncaught_count)
        , m_active(other.m_active)
    {
        other.m_active = false;
    }

    ScopeGuardOnFail& operator=(ScopeGuardOnFail&& other) noexcept(std::is_nothrow_move_assignable_v<F>) {
        if (this != &other) {
            // Execute if active and unwinding
            if (m_active && std::uncaught_exceptions() > m_uncaught_count) {
                try { m_action(); } catch (...) {}
            }
            m_action = std::move(other.m_action);
            m_uncaught_count = other.m_uncaught_count;
            m_active = other.m_active;
            other.m_active = false;
        }
        return *this;
    }

    ScopeGuardOnFail(const ScopeGuardOnFail&) = delete;
    ScopeGuardOnFail& operator=(const ScopeGuardOnFail&) = delete;

    ~ScopeGuardOnFail() noexcept {
        // Only execute if more exceptions are in flight than at construction
        if (m_active && std::uncaught_exceptions() > m_uncaught_count) {
            try { m_action(); } catch (...) {}
        }
    }

    void dismiss() noexcept { m_active = false; }
    [[nodiscard]] bool is_active() const noexcept { return m_active; }

private:
    F m_action;
    int m_uncaught_count;
    bool m_active;
};

// =============================================================================
// ScopeGuardOnSuccess - Executes Only on Normal Exit
// =============================================================================

/**
 * @brief Scope guard that executes only when leaving scope normally (no exception).
 * 
 * Perfect for implementing commit semantics in transactions.
 * 
 * @tparam F The type of the cleanup function object
 * 
 * Example:
 * @code
 * void process_transaction() {
 *     begin_transaction();
 *     auto commit = makeScopeGuardOnSuccess([&]() { commit_transaction(); });
 *     auto rollback = makeScopeGuardOnFail([&]() { rollback_transaction(); });
 *     do_work();  // If this throws, rollback runs; if not, commit runs
 * }
 * @endcode
 */
template <typename F>
class [[nodiscard]] ScopeGuardOnSuccess {
public:
    explicit ScopeGuardOnSuccess(F&& action) noexcept(std::is_nothrow_move_constructible_v<F>)
        : m_action(std::forward<F>(action))
        , m_uncaught_count(std::uncaught_exceptions())
        , m_active(true)
    {}

    ScopeGuardOnSuccess(ScopeGuardOnSuccess&& other) noexcept(std::is_nothrow_move_constructible_v<F>)
        : m_action(std::move(other.m_action))
        , m_uncaught_count(other.m_uncaught_count)
        , m_active(other.m_active)
    {
        other.m_active = false;
    }

    ScopeGuardOnSuccess& operator=(ScopeGuardOnSuccess&& other) noexcept(std::is_nothrow_move_assignable_v<F>) {
        if (this != &other) {
            // Execute if active and NOT unwinding
            if (m_active && std::uncaught_exceptions() == m_uncaught_count) {
                try { m_action(); } catch (...) {}
            }
            m_action = std::move(other.m_action);
            m_uncaught_count = other.m_uncaught_count;
            m_active = other.m_active;
            other.m_active = false;
        }
        return *this;
    }

    ScopeGuardOnSuccess(const ScopeGuardOnSuccess&) = delete;
    ScopeGuardOnSuccess& operator=(const ScopeGuardOnSuccess&) = delete;

    ~ScopeGuardOnSuccess() noexcept {
        // Only execute if same number of exceptions as at construction (normal exit)
        if (m_active && std::uncaught_exceptions() == m_uncaught_count) {
            try { m_action(); } catch (...) {}
        }
    }

    void dismiss() noexcept { m_active = false; }
    [[nodiscard]] bool is_active() const noexcept { return m_active; }

private:
    F m_action;
    int m_uncaught_count;
    bool m_active;
};

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * @brief Create a ScopeGuard with the default policy (terminate on exception).
 * 
 * @tparam F The cleanup function type
 * @param fn The cleanup function
 * @return ScopeGuard that will execute fn on scope exit
 * 
 * Example:
 * @code
 * auto guard = makeScopeGuard([&]() noexcept { cleanup(); });
 * @endcode
 */
template <typename F>
[[nodiscard]] auto makeScopeGuard(F&& fn) 
    noexcept(std::is_nothrow_constructible_v<ScopeGuard<std::decay_t<F>>, F&&>)
{
    return ScopeGuard<std::decay_t<F>>(std::forward<F>(fn));
}

/**
 * @brief Create a ScopeGuard with a specific policy.
 * 
 * @tparam Policy The exception handling policy
 * @tparam F The cleanup function type
 * @param fn The cleanup function
 * @return ScopeGuard with the specified policy
 * 
 * Example:
 * @code
 * auto guard = makeScopeGuard<ScopeGuardSwallowPolicy>([&]() { risky_cleanup(); });
 * @endcode
 */
template <typename Policy, typename F>
[[nodiscard]] auto makeScopeGuard(F&& fn)
    noexcept(std::is_nothrow_constructible_v<ScopeGuard<std::decay_t<F>, Policy>, F&&>)
{
    return ScopeGuard<std::decay_t<F>, Policy>(std::forward<F>(fn));
}

/**
 * @brief Create a ScopeGuardOnFail (executes only on exception).
 */
template <typename F>
[[nodiscard]] auto makeScopeGuardOnFail(F&& fn)
    noexcept(std::is_nothrow_constructible_v<ScopeGuardOnFail<std::decay_t<F>>, F&&>)
{
    return ScopeGuardOnFail<std::decay_t<F>>(std::forward<F>(fn));
}

/**
 * @brief Create a ScopeGuardOnSuccess (executes only on normal exit).
 */
template <typename F>
[[nodiscard]] auto makeScopeGuardOnSuccess(F&& fn)
    noexcept(std::is_nothrow_constructible_v<ScopeGuardOnSuccess<std::decay_t<F>>, F&&>)
{
    return ScopeGuardOnSuccess<std::decay_t<F>>(std::forward<F>(fn));
}

// =============================================================================
// Macro Support Infrastructure
// =============================================================================

namespace detail {

// Maker structs for operator+ trick used by macros
struct ScopeGuardMaker {};
struct ScopeGuardOnFailMaker {};
struct ScopeGuardOnSuccessMaker {};

template <typename Fn>
[[nodiscard]] ScopeGuard<std::decay_t<Fn>> operator+(ScopeGuardMaker, Fn&& fn) {
    return ScopeGuard<std::decay_t<Fn>>(std::forward<Fn>(fn));
}

template <typename Fn>
[[nodiscard]] ScopeGuardOnFail<std::decay_t<Fn>> operator+(ScopeGuardOnFailMaker, Fn&& fn) {
    return ScopeGuardOnFail<std::decay_t<Fn>>(std::forward<Fn>(fn));
}

template <typename Fn>
[[nodiscard]] ScopeGuardOnSuccess<std::decay_t<Fn>> operator+(ScopeGuardOnSuccessMaker, Fn&& fn) {
    return ScopeGuardOnSuccess<std::decay_t<Fn>>(std::forward<Fn>(fn));
}

} // namespace detail

// =============================================================================
// Convenience Macros
// =============================================================================

// Unique name generation
#if defined(__COUNTER__)
#define TF_SCOPE_GUARD_CONCAT_IMPL(a, b) TF_SCOPE_GUARD_CONCAT_IMPL2(a, b)
#define TF_SCOPE_GUARD_CONCAT_IMPL2(a, b) a##b
#define TF_SCOPE_GUARD_UNIQUE(prefix) TF_SCOPE_GUARD_CONCAT_IMPL(prefix, __COUNTER__)
#else
#define TF_SCOPE_GUARD_CONCAT_IMPL(a, b) TF_SCOPE_GUARD_CONCAT_IMPL2(a, b)
#define TF_SCOPE_GUARD_CONCAT_IMPL2(a, b) a##b
#define TF_SCOPE_GUARD_UNIQUE(prefix) TF_SCOPE_GUARD_CONCAT_IMPL(prefix, __LINE__)
#endif

/**
 * @brief Macro for scope guard that always executes.
 * 
 * Usage: TF_SCOPE_EXIT { cleanup_code; };
 * 
 * @warning 'return' inside the block returns from the lambda, NOT the function!
 * 
 * Example:
 * @code
 * TF_Tensor* tensor = TF_AllocateTensor(...);
 * TF_SCOPE_EXIT { TF_DeleteTensor(tensor); };
 * // tensor is automatically deleted when scope exits
 * @endcode
 */
#define TF_SCOPE_EXIT \
    auto TF_SCOPE_GUARD_UNIQUE(tf_scope_exit_) = ::tf_wrap::detail::ScopeGuardMaker{} + [&]() noexcept

/**
 * @brief Macro for scope guard that executes only on exception.
 * 
 * Usage: TF_SCOPE_FAIL { rollback_code; };
 * 
 * @warning 'return' inside the block returns from the lambda, NOT the function!
 * 
 * Example:
 * @code
 * allocate_resource_a();
 * TF_SCOPE_FAIL { release_resource_a(); };
 * allocate_resource_b();  // If this throws, resource_a is released
 * @endcode
 */
#define TF_SCOPE_FAIL \
    auto TF_SCOPE_GUARD_UNIQUE(tf_scope_fail_) = ::tf_wrap::detail::ScopeGuardOnFailMaker{} + [&]() noexcept

/**
 * @brief Macro for scope guard that executes only on success (no exception).
 * 
 * Usage: TF_SCOPE_SUCCESS { commit_code; };
 * 
 * @warning 'return' inside the block returns from the lambda, NOT the function!
 * 
 * Example:
 * @code
 * begin_transaction();
 * TF_SCOPE_SUCCESS { commit(); };
 * TF_SCOPE_FAIL { rollback(); };
 * do_work();  // commit on success, rollback on exception
 * @endcode
 */
#define TF_SCOPE_SUCCESS \
    auto TF_SCOPE_GUARD_UNIQUE(tf_scope_success_) = ::tf_wrap::detail::ScopeGuardOnSuccessMaker{} + [&]() noexcept

} // namespace tf_wrap
