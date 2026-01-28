**UNCLASSIFIED** 
**Migration Checklist — C → Modern C++**  
*(Combined original steps and detailed architectural analysis)*  

---  

## I. Strategic Execution Checklist  

### Part A: System‑Wide Policies (establish once before migration begins)

| # | Decision / Action | Guidance / Rationale |
|---|-------------------|----------------------|
| **A1** | **Namespace convention** | Define the hierarchy pattern (`company::project::module`). Modules will be decomposed into this structure. |
| **A2** | **Error‑handling policy** | Choose one approach and apply consistently across all modules: <br>• Return codes (legacy compatibility). <br>• Exceptions (programming/invariant violations). <br>• `std::expected` (recoverable business errors). Use `[[nodiscard]]` to force callers to check results. |
| **A3** | **Contract / invariant approach** | Establish Design‑by‑Contract conventions: `static_assert` for compile‑time checks, runtime `Expects/Ensures` (e.g., GSL) for debugging builds, class‑invariant checks in destructors. |
| **A4** | **Tooling & verification** | Integrate clang‑tidy, clang‑format, AddressSanitizer, ThreadSanitizer, and a CI pipeline that builds on GCC, Clang, and MSVC. |
| **A5** | **CI/CD pipeline** | Build, static analysis, sanitizers, nightly fuzzing of the C API, and automated deployment of test artefacts. |
| **A6** | **Testing framework** | Establish patterns for unit tests (GoogleTest), integration tests that call the C façade, property‑based tests for error‑code translation, and benchmark regression tests. |
| **A7** | **Documentation templates** | Produce an API‑diff document template, ABI stability policy template, and module migration record format. |
| **A8** | **Avoidances** | • Premature optimisation — profile first.<br>• Unnecessary fancy abstractions.<br>• C‑style casts — use `static_cast`, `reinterpret_cast`, `const_cast`.<br>• "God" service locators — split by subsystem.<br>• Raw `new`/`delete` in files that still include old C headers. |

### Part B: Per‑Module Workflow (apply when migrating each module)

| # | Decision / Action | Guidance / Rationale |
|---|-------------------|----------------------|
| **B1** | **Define C‑API & ABI boundaries** | Identify this module's public `extern "C"` entry points that must stay ABI‑stable. Mark with versioned macros per system policy. |
| **B2** | **Classify globals & pick dependency strategy** | For *this module*, categorize globals as:<br>• **Singletons** — immutable/read‑only objects (e.g., hardware maps).<br>• **Service Locator** — when avoiding signature churn is paramount.<br>• **Dependency Injection** — when explicit constructor wiring is worthwhile.<br>Choose based on the module's use case. |
| **B3** | **Decompose into namespace hierarchy** | Map this module's procedural logic into the system namespace convention. Eliminates name collisions before any class is written. |
| **B4** | **Concurrency requirements** | Define *ownership flow* for this module (who creates, who destroys, who shares). Choose a model: **Message Passing**, **Shared State with locks**, or **Lock‑free**. |
| **B5** | **Memory & RAII strategy** | Decide ownership semantics (`unique_ptr` for exclusive transfer, `shared_ptr` only when true sharing is needed, custom object pools for high‑throughput buffers). Wrap every resource in an RAII wrapper. |
| **B6** | **Select language features** | Only after B4–B5 pick concrete C++ facilities (`std::vector`, `std::span`, `std::string_view`, atomics, `std::ranges`, etc.). |
| **B7** | **Write tests** | Implement unit tests for the module, integration tests against the C façade, and property‑based tests for error‑code translation. |
| **B8** | **Performance regression loop** | Establish a baseline C benchmark for this module. After migration, run the same benchmark and enforce a "no slowdown > 5 %" rule before merging. |
| **B9** | **Update documentation** | Update API‑diff, record module‑specific ownership/concurrency/dependency decisions, and keep the design spec current. |

---  

## II. Detailed Architectural Analysis & Justification  

### 1. Service Locator vs. Dependency Injection  

| Aspect | Dependency Injection (DI) | Service Locator (SL) |
|--------|---------------------------|----------------------|
| **Signature bloat** | Requires every dependent object to be passed through constructors → many‑parameter functions. | One global access point (`Locator::get<T>()`) avoids touching every signature. |
| **Ownership & lifetime** | Forces explicit ownership decisions; often leads to `shared_ptr` overhead. | Locator owns services (usually `unique_ptr` or static objects) and controls shutdown order. |
| **Performance/indirection** | Relies on virtual interfaces → v‑table look‑ups, hindering inlining. | Can return concrete references, eliminating virtual dispatch for hot paths. |
| **Ravioli code trap** | Over‑decomposition creates many tiny classes whose interaction graph becomes hard to trace. | Keeps the high‑level dependency graph flat; static analysis can still enumerate all `Locator::get<T>()` calls. |
| **Test/config substitution** | Swapping implementations often needs recompilation of the injector. | Implementations can be substituted for tests or configured at startup without threading dependencies through every signature. |
| **Visibility** | Dependencies are explicit in constructors, but the sheer number of parameters can obscure intent. | Locator calls are auditable via code search and tooling (clang‑tidy, grep). |

**Takeaway:** For a large legacy C codebase, a **Service Locator** is usually the pragmatic first step. DI can be introduced later for new, self‑contained modules where constructor‑level wiring is acceptable.

---

### 2. Why Concurrency Design Precedes RAII  

1. **Ownership semantics are dictated by the concurrency model** — e.g., message‑passing → `unique_ptr`; shared state → `shared_ptr` or lock‑protected raw pointers.  
2. **Destruction safety** — Without a clear concurrency model you may destroy an object while another thread still holds a reference, leading to use‑after‑free bugs.  
3. **Lock‑management needs context** — You can't write generic `LockGuard<T>` until you know which resources need protection, the lock hierarchy, and dead‑lock avoidance strategy.  
4. **Performance implications** — Picking the wrong RAII guard (e.g., a mutex guard for a lock‑free path) creates contention that is far harder to refactor later.  

> **Conclusion:** Design the ownership flow, thread‑communication pattern, and synchronization strategy *first*, then create RAII wrappers that precisely match those decisions.

---

### 3. Why Memory / RAII Precedes Feature Selection  

| Pitfall | What Happens When Feature Is Chosen Too Early |
|---------|-----------------------------------------------|
| **`std::vector` in a shared‑state module** | Later you discover the container must be immutable → you end up locking every access (contention) or copying the vector per thread (excessive allocations). |
| **`std::string` in a real‑time path** | Hidden heap allocations cause jitter; the proper solution (fixed‑size buffer) was missed because the type was selected first. |
| **Object pool after RAII** | Scattered `new/delete` calls become hard to replace; once you centralise allocation, you can swap in a custom pool without touching high‑level logic. |

**Strategy:**  
1. Define ownership flow (who creates, who destroys, who shares).  
2. Choose the appropriate smart‑pointer / container policy based on that flow.  
3. Only then pick convenience features (`std::span`, `std::ranges`, etc.).  

---

### 4. Validation & Performance Loop  

| Phase | Goal | Tools / Practices |
|-------|------|-------------------|
| **Unit Testing** | Verify each isolated C++ module. | GoogleTest (GTest) — use the Service Locator to inject mocks. |
| **Integration Testing** | Ensure the C façade forwards correctly to the new implementation. | Build test harnesses that call the original C headers and compare results. |
| **Benchmarking** | Detect performance regressions early. | Google Benchmark — keep a baseline on the original C functions; enforce "no slowdown > 5 %". |
| **Static Analysis** | Catch common C→C++ pitfalls (raw pointers, C‑style casts, mismatched `new/delete`). | clang‑tidy, cppcheck, clang‑static‑analyzer. |
| **Dynamic Sanitizers** | Find memory‑corruption and data‑race bugs introduced during migration. | AddressSanitizer (ASan), ThreadSanitizer (TSan), UndefinedBehaviorSanitizer (UBSan). |
| **Continuous Integration** | Enforce that every commit passes the above checks on multiple platforms. | GitHub Actions / GitLab CI, matrix builds with GCC, Clang, MSVC. |
| **Fuzzing** | Guard against malformed input crossing the C‑API boundary. | libFuzzer or AFL targeting exported `extern "C"` functions. |

**Performance Regression Loop Example**

```cpp
// legacy.c
int process_packet(const uint8_t *buf, size_t len);

// modern.cpp
std::expected<Packet, Error> process_packet(std::span<const uint8_t> data);
```

1. Run `benchmark_legacy` → obtain *X ns/op*.  
2. Run `benchmark_modern` → obtain *Y ns/op*.  
3. Assert `Y <= X * 1.05`.  
4. If the assertion fails, profile (`perf`, `VTune`) and decide whether to redesign the abstraction or accept the trade‑off.

---

## III. Final‑Mile Tactics (The "No Fancy" Rule)  

| Tactic | Rationale |
|--------|-----------|
| **Avoid "God" Locators** | Split the locator by subsystem (`NetLocator`, `DiskLocator`, `UiLocator`). Keeps each small, testable, and easier to reason about. |
| **Transitionary Shims** | Create C‑style wrapper functions that simply forward to the new C++ implementation (e.g., `int old_api_do_work(void)` → `return wrap_do_work();`). Enables incremental migration without breaking existing callers. |
| **Zero‑Cost Views** | Prefer `std::span` and `std::string_view` for read‑only buffer access — no allocation, bounds‑checked when compiled with sanitizers. |
| **Explicit Casts Only** | Replace every C‑style cast with `static_cast`, `reinterpret_cast`, or `const_cast`. Makes intent searchable and prevents accidental reinterpretations. |
| **Avoid Hidden Heap Allocation** | When the size is known at compile time, use `std::array` or a custom ring buffer instead of `std::vector`. |
| **Prefer `auto` When Unambiguous** | Use `auto` for iterator types or lambdas, but keep explicit return types for public interfaces so callers know the exact type. |
| **Layered Builds** | Keep the C façade in a separate static library that depends on the modern C++ core library. This isolates legacy build configuration and makes it easy to drop the C layer later. |
| **Document ABI Guarantees** | Maintain an `ABI.md` listing every exported symbol, its version, and the stability contract ("stable for 3 major releases"). |
| **Profile‑First, Optimize‑Later** | Use `perf`, `gprof`, or `VTune` on real workloads before hand‑optimising any abstraction. |
| **Code‑Review Checklist** | Add a short checklist to PR templates: <br>• No raw `new`/`delete` (RAII only). <br>• No C‑style casts. <br>• All public functions are `[[nodiscard]]` where appropriate. <br>• Error handling matches the chosen policy. |

---

## IV. Bottom Line  

By **keeping structural design decisions** (API/ABI definition, globals handling, namespace layout, error policy, contracts, ownership, concurrency, and feature selection) **ahead of syntactic implementation**, you ensure a migration that yields a **genuinely modern C++** system rather than "C disguised as C++."  

The combined checklist and analysis give you:

* A clear **two‑phase structure**: system‑wide policies/infrastructure (Part A) locked in once, then a repeatable per‑module workflow (Part B) applied to each unit of work.  
* A **justified preference** for Service Locator over full DI in many legacy contexts—while recognizing that SL vs DI is a per‑module judgment call, not a blanket policy.  
* The reasoning why **concurrency → RAII → feature choice** is the correct flow.  
* A **validation & performance loop** to catch regressions early.  
* Concrete **final‑mile tactics** to enforce the "no fancy unless it adds value" principle.  

Follow the steps, let the tooling enforce the discipline, and you'll end up with a clean, maintainable, high‑performance Modern C++ codebase ready for future evolution.
