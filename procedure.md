Strategic Roadmap: C to Modern C++ Migration
This document establishes the definitive architectural sequence for migrating a legacy C codebase to Modern C++. It prioritizes Structural Design Decisions over Syntactic Implementation to ensure the result is a maintainable C++ system rather than "C disguised as C++."
I. The Strategic Execution Checklist
Define C-API & ABI Boundaries: Secure the perimeter with extern "C". Identify which boundaries must remain ABI-invariant to ensure the system remains linkable with legacy C components.
Classify Globals & Dependency Strategy:
Singletons: Reserved for immutable/read-only objects (e.g., fixed hardware maps).
Service Locator: The primary solution for mutable/extensible services. It manages system context without "signature bloat" or "dependency hell."
Dependency Injection: Reserved for tight, local couplings where visibility is more critical than API stability.
Logical Breakup & Namespacing: Deconstruct procedural logic into a namespace hierarchy. This defines the structural "map" of the module and solves global name collisions before a single class is written.
Error Handling Policy: Standardize on std::expected for logic failures and Exceptions for system/invariant violations. Use [[nodiscard]] to force caller acknowledgement.
Contract Enforcement: Implement "Design by Contract" via class invariants and move runtime checks to compile-time using static_assert.
Concurrency Model: Define the "Ownership Flow" (Who creates? Who destroys? Is it shared?). Determine if the model is based on Message Passing, Shared State, or Monitor Objects.
Memory & RAII Design: Based on the concurrency flow, determine ownership semantics (e.g., unique_ptr for transfer vs. shared_ptr for shared access).
Feature Selection (Implementation): Only now do you select specific C++ features (e.g., std::vector, std::string, std::atomic). This ensures the "vehicle" fits the "road" designed in the previous steps.
Tooling & Verification: Integrate AddressSanitizers (ASan) and Clang-Tidy to catch legacy habits and memory bugs.
II. Detailed Architectural Analysis & Justification
1. The Case for the Service Locator vs. Dependency Injection
While Dependency Injection (DI) is a cornerstone of modern software, its implementation in large, legacy C codebases presents unique and significant challenges that often make the Service Locator a more pragmatic first step.
The Problem of "Constructor Madness": In large C codebases, dependencies are often global or deeply nested. Pure DI requires every dependency to be passed through constructors or function arguments. If a low-level utility (e.g., a logger) is needed 15 levels deep, you must refactor every function signature in that chain, leading to "signature bloat" where constructors take 10+ arguments.
Ownership and Lifetime Complexity: C memory is typically manual or static. DI shifts ownership to the "injector." This often causes "shared_ptr infection," where developers use atomic reference counting to solve lifetime mismatches, introducing performance overhead and making it difficult to reason about when memory is actually freed.
Performance and Indirection: DI typically relies on interfaces (pure virtual classes) to allow for swapping implementations. This introduces virtual function tables (vtables) and pointer indirection, which can hinder compiler optimizations like inlining—a critical concern in high-performance C systems.
The "Ravioli Code" Trap: Refactoring legacy "spaghetti" (tightly coupled functions) into DI can result in "ravioli code"—thousands of tiny, decoupled classes. While individual units are simple, the interactions between them become a complex web that is nearly impossible to trace without specialized tools.
The Locator as a "Runtime Linker": The Service Locator avoids these pitfalls by allowing you to "teleport" dependencies exactly where needed. It maintains a stable public API while allowing you to swap a "Production Hardware Driver" for a "Software Simulator" in a single configuration line.
Static Analysis & Visibility: Critics claim locators hide dependencies. However, modern Clang-Tidy tools can index Locator::get<T>() calls to generate a full dependency graph. This provides better visibility than a 15-argument constructor where dependencies are lost in the noise. 
2. Why Concurrency Design Precedes RAII
RAII (Resource Acquisition Is Initialization) is the implementation of Ownership and Lifetime. You cannot implement it until you have designed the Concurrency Model, because concurrency dictates who owns the resource and how long it must live.
Ownership Semantics: If your concurrency model is Message Passing, your RAII strategy will be std::unique_ptr to move ownership between threads. If you use Shared State, you are forced into std::shared_ptr. Choosing the wrong one early causes massive refactors.
Destruction Safety: Concurrency determines if an object can be safely destroyed at the end of a local scope or if its lifetime must be extended non-deterministically across threads.
Lock Management: You cannot implement RAII Lock Guards until you know which resources need protection and where the critical sections are. Designing RAII before Concurrency leads to deadlocks or "use-after-free" bugs when destructors trigger while other threads are still active.
3. Why Memory/RAII Precedes Feature Selection
"Feature-first" implementation (using std::vector or std::string) before defining Memory Ownership results in "C with Classes."
The Trap: If you implement a module with std::vector and later realize the Concurrency Design requires it to be immutable or shared, you end up "locking the container" at every access point—a performance anti-pattern.
The Solution: By defining Memory Ownership first (e.g., "This buffer is owned by the Network thread and moved to the Logic thread"), you realize that a std::span or a custom circular buffer is a better choice than a generic std::vector.
Strategic Feature Selection: Features should be the "vehicle" chosen to fit the "road" (Concurrency) and the "cargo" (Memory RAII). This prevents over-allocating memory just because "that's how std::string works."
III. Validation & Performance Loop
Unit Testing throughout: Use GoogleTest (GTest). The Service Locator facilitates mocking, allowing you to write modern tests for migrated logic even while it remains surrounded by legacy C code.
Iterative Benchmarking: C++ abstractions are only "zero-cost" if used correctly. Use Google Benchmark for every module converted to ensure modernization doesn't introduce a performance hit.
Sanitizers: Running AddressSanitizer (ASan) and ThreadSanitizer (TSan) is non-negotiable to catch buffer overflows and data races where C-logic and C++ objects first overlap.
IV. Final-Mile Tactics (The "No Fancy" Rule)
Avoid "God Locators": Segregate locators by subsystem (e.g., NetLocator, DiskLocator) to prevent them from becoming monolithic.
Transitionary Shims: Create C++ wrappers that look like old C functions. Inside, call the new C++ logic via the Locator. This allows for an Incremental Migration where internals change without breaking the system's external view.
Zero-Cost Views: Favor std::span and std::string_view for bounds-checked safety without the performance cost of copying memory.
Avoid C-Style Casts: Use static_cast (safe) or reinterpret_cast (explicit) to make intent clear and searchable.
