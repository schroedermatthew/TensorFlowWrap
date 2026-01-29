# Migration Procedure — C to Modern C++

## Executive Summary

This procedure defines how we migrate legacy C code to modern C++ without breaking existing systems or creating rework cascades. The key insight is that **design decisions must be locked in a specific order**—boundaries before dependencies, concurrency before memory management, ownership before feature selection—because reversing that order forces expensive revisits across already-migrated code.

The checklists below capture both the system-wide policies we establish once and the repeatable workflow we apply to each module. Part 2 provides definitions, justifications, and an analysis of how this procedure increases development velocity.

---

## Part 1 — Strategic Execution Checklist

### Part A: System-Wide Policies

Establish these decisions once, before any module migration begins.

| #      | Policy                          | Summary                                                                 |
|--------|---------------------------------|-------------------------------------------------------------------------|
| **A1** | Namespace convention            | Define the naming hierarchy all modules will use.                       |
| **A2** | Error-handling policy           | Standardize how functions report success and failure in C++ code.       |
| **A3** | C boundary translation          | Define how errors and exceptions convert to stable C return codes.      |
| **A4** | C ABI contract                  | Hard rules for what can and cannot cross the C interface.               |
| **A5** | Contract and invariant approach | Establish how preconditions and postconditions are checked.             |
| **A6** | Tooling and verification        | Specify static analysis, sanitizers, and CI requirements.               |
| **A7** | Testing patterns                | Define categories of tests every module must have.                      |
| **A8** | Avoidances                      | Explicit bans on patterns that create maintenance problems.             |

### Part B: Per-Module Workflow

Apply these steps in order when migrating each module.

| #      | Step                                | Summary                                                                  |
|--------|-------------------------------------|--------------------------------------------------------------------------|
| **B1** | Define C API and ABI boundaries     | Identify what external callers depend on and what must stay stable.      |
| **B2** | Classify globals and dependencies   | Decide how the module obtains the services and data it needs.            |
| **B3** | Decompose into namespace hierarchy  | Organize code into the system naming convention before adding types.     |
| **B4** | Define concurrency requirements     | Declare threading behavior and shutdown rules.                           |
| **B5** | Decide memory and ownership         | Define who creates, owns, and destroys each resource.                    |
| **B6** | Identify compile-time opportunities | Determine which computations, checks, and configurations can be resolved at build time. |
| **B7** | Select language features            | Choose C++ facilities that match the ownership, concurrency, and compile-time models. |
| **B8** | Write tests                         | Verify correctness of both the C++ core and the C interface.             |
| **B9** | Validate performance                | Confirm the migrated code meets performance requirements.                |
| **B10** | Update documentation                | Record decisions so future work does not contradict them.                |

---

## Part 2 — Definitions and Justification

### Why This Order Matters

Migration failures rarely come from choosing the wrong C++ feature. They come from making decisions in the wrong order, which forces teams to revisit already-migrated code when a later decision invalidates earlier assumptions.

The ordering in Part B prevents this by ensuring each decision is informed by—and constrained by—the decisions before it:

1. **Boundaries first** because they define what you cannot change without breaking external systems. Every later decision must fit within what the boundary can safely represent.

2. **Dependencies second** because how a module obtains services determines how invasive changes will be. Changing dependency strategy late means touching every function that uses those services.

3. **Namespaces third** because organizing names and groupings creates seams for later refactoring with minimal risk. It is cheap to do early and expensive to do after types and classes exist.

4. **Concurrency fourth** because threading behavior dictates ownership rules. You cannot correctly define who owns an object until you know who can access it concurrently.

5. **Ownership fifth** because memory management must encode the concurrency and lifetime rules already established. Choosing smart pointers or object pools before understanding threading leads to either race conditions or unnecessary synchronization.

6. **Compile-time identification sixth** because the previous steps reveal which values are fixed by design versus genuinely dynamic. This classification must happen before selecting language features, since compile-time and runtime computation use different facilities.

7. **Features last** because containers, views, and utilities must match the ownership model and compile-time/runtime classification. Selecting a container first and then discovering it does not fit the concurrency requirements forces redesign. Using runtime computation for compile-time-known values wastes performance and delays error detection.

---

### How This Procedure Increases Velocity

Migration projects fail not because teams lack C++ expertise but because rework consumes the schedule. A decision made in week three invalidates work done in weeks one and two. Developers revisit the same files repeatedly. Merge conflicts multiply. Estimates become meaningless.

This procedure is designed to eliminate rework by front-loading decisions that constrain later choices. Each step produces a stable foundation that subsequent steps build on without disturbing. The result is that developers move forward, not in circles.

**Stable boundaries enable parallel work.** Once B1 defines what external callers depend on, internal restructuring cannot break them. Teams working on different modules can proceed independently because the contract between modules is fixed. Changes inside a module do not ripple into other teams' code.

**Service registries reduce signature churn.** The choice in B2 between a service registry and explicit parameter passing has a direct impact on changeset size. With explicit passing, adding a new dependency to a low-level function requires changing its signature—and the signature of every function that calls it, up to the entry point. A single new dependency can touch dozens of files, increasing review burden and merge conflict risk. With a service registry, adding a dependency means registering it once and requesting it where needed. Files that do not use the new dependency remain untouched. New code can be written immediately without first solving how to route access through the call graph.

**Early namespace structure prevents reorganization churn.** B3 establishes where code lives before classes and inheritance relationships exist. Moving a free function between namespaces is trivial. Moving a class that other classes inherit from, or that templates specialize on, requires updating every dependent. Doing the organizational work early, when the cost is low, avoids doing it late when the cost is high.

**Upfront concurrency decisions prevent ownership rework.** B4 locks in threading behavior before B5 chooses ownership patterns. If ownership is decided first and threading requirements emerge later, the ownership model often must be revised—shared pointers added where exclusive ownership was assumed, synchronization retrofitted into types that were designed without it. Each revision touches files that were already reviewed and tested. Deciding concurrency first means ownership decisions stick.

**Compile-time classification accelerates feedback loops.** B6 identifies which checks and computations can happen at build time. Errors caught by the compiler appear in seconds; errors caught at runtime require building, deploying, running, and debugging. Shifting validation earlier in the development cycle means developers learn about problems faster and fix them before the code leaves their workstation.

**Deferred feature selection avoids premature commitment.** B7 comes late precisely so that earlier decisions are not revisited when feature limitations surface. Choosing a container before understanding ownership leads to fighting the container's semantics—refactoring to a different container, updating call sites, re-running tests. Choosing after ownership and concurrency are settled means the choice fits the first time.

**Per-module testing catches defects early.** B8 validates each module before integration. A defect found during module testing is localized; a defect found after integration with other modules requires determining which module is at fault, coordinating fixes across teams, and retesting the integration. Early testing keeps debugging local and fast.

**Per-module performance validation prevents late-stage crises.** B9 compares each module against its baseline before the module is declared complete. Performance problems discovered late—after multiple modules are integrated—are difficult to attribute and expensive to fix because the design is already frozen. Catching regressions per-module keeps options open and fixes cheap.

**Documentation preserves decisions.** B10 records what was decided and why. Without this, future developers re-examine settled questions, sometimes reaching different conclusions that contradict the existing design. The result is inconsistency, confusion, and rework to reconcile the contradiction. Documentation lets future work build on past decisions rather than relitigate them.

**The ordering itself is a velocity multiplier.** The sequence from B1 to B10 is not arbitrary. Each step depends on the stability of the steps before it. When the ordering is followed, each decision is made once and remains valid. When the ordering is violated—when features are chosen before ownership, or ownership before concurrency—later discoveries invalidate earlier work, and the team pays for the same decision multiple times.

The goal is not just correct code. It is correct code delivered on schedule, with predictable progress, and without the morale damage that comes from repeated rework. This procedure achieves that by making the expensive decisions early, when changing them is cheap, and protecting them from disruption as the migration proceeds.

---

### A1 — Namespace Convention

**What it means:** A namespace is a named scope that groups related code and prevents name collisions. The convention defines a consistent hierarchy pattern such as `company::project::module` that all migrated code follows.

**Why it matters:** Legacy C code often uses prefixes (like `proj_module_function`) to avoid collisions. Namespaces replace this convention with language-enforced scoping, but only if all teams use the same structure. Inconsistent naming creates confusion about where code belongs and makes refactoring harder.

**What to decide:** The hierarchy depth, the naming rules for each level, and whether certain names are reserved for specific purposes.

---

### A2 — Error-Handling Policy

**What it means:** Functions must communicate success or failure to their callers. The policy standardizes which mechanism functions use and when.

**The two mechanisms:**

- **Expected values:** A function returns either the successful result or an error description, packaged together. The caller must inspect the return to determine which occurred. This is appropriate when failure is a normal, anticipated outcome—such as a file not being found, input failing validation, or a network timeout.

- **Exceptions:** A function signals failure by throwing, which transfers control up the call stack until something catches it. This is appropriate for failures that are not normal outcomes—such as violations of internal assumptions, construction that cannot complete, or situations where continuing would be meaningless.

**The rule:** Each function chooses one mechanism. A function either returns an expected value (and does not throw for normal failures) or throws (and returns a plain result on success). Mixing both mechanisms in the same function creates ambiguity about what callers must handle.

**Why it matters:** Inconsistent error handling forces callers to guess. If some functions throw and others return error values with no clear pattern, every call site must defensively handle both possibilities, which adds complexity and invites mistakes.

---

### A3 — C Boundary Translation

**What it means:** The C interface is the stable surface that existing systems call. It cannot use C++ error mechanisms directly. This policy defines how errors inside the C++ implementation become error codes that C callers understand.

**The hard rule:** Exceptions never cross the C boundary. If the C++ implementation throws, the C interface layer must catch that exception and convert it to an error code before returning.

**Translation approach:**

- If the C++ function returns an expected value, the interface layer inspects it: success populates the output parameters and returns an "OK" code; failure maps the error to a documented code.

- If the C++ function throws, the interface layer catches it: known exception types map to specific codes; unknown exceptions map to a general "unexpected error" code.

**Optional enhancement:** Store additional diagnostic information (such as an error message or context) in thread-local storage. Provide a separate C function that callers can use to retrieve this information when debugging, without changing the primary return-code interface.

**Why it matters:** Exceptions escaping across library boundaries cause crashes or undefined behavior. This rule makes the boundary safe and predictable.

---

### A4 — C ABI Contract

**What it means:** The Application Binary Interface is the low-level contract that allows compiled code to interoperate. This policy defines the rules that keep the C interface stable across compiler versions, platforms, and library updates.

**The rules:**

- **No C++ types in the interface.** Standard library types like strings, vectors, and containers cannot appear in function signatures or structures exposed to C. Their binary layout is not guaranteed stable.

- **Separate C-visible headers from C++ implementation headers.** The headers that C code includes must not transitively include any C++ headers. A common migration mistake is including a C++ implementation header from a C-visible header, which breaks compilation for all C consumers.

- **Use opaque handles for objects.** When C code must hold a reference to a C++ object, expose it as a pointer to an incomplete type. Provide explicit create and destroy functions. This hides the object's size and layout from callers.

- **Use fixed-width integers.** Types like `int` and `long` vary in size across platforms. Use explicitly-sized types (32-bit integers, 64-bit integers) in interface signatures to guarantee consistent behavior.

- **Define allocation ownership.** For every buffer or object that crosses the boundary, document whether the caller or the implementation is responsible for freeing it. When the implementation allocates, provide a dedicated free function rather than expecting callers to use their own allocator.

- **Version structures if they must change.** If a structure crosses the boundary and might grow in future versions, include a size or version field so code can detect mismatches.

**Why it matters:** ABI violations cause subtle, hard-to-diagnose failures—crashes, corruption, or wrong results that only appear in specific configurations. These rules prevent them.

---

### A5 — Contract and Invariant Approach

**What it means:** Contracts are documented expectations about what must be true when a function is called (preconditions) and what will be true when it returns (postconditions). Invariants are properties that must always hold for an object to be in a valid state.

**How to enforce them:**

- **Compile-time checks** verify conditions that can be evaluated when code is compiled. Use these for constraints on types, sizes, or configurations.

- **Runtime checks** verify conditions during execution. Enable these in development and testing builds to catch violations early. The policy must specify whether these remain active in release builds or are removed for performance.

**Where to check invariants:** Check at public function boundaries—when entering and before returning. Be cautious about checking invariants in destructors, since during teardown an object may be intentionally partially dismantled.

**Why it matters:** Contracts make assumptions explicit. When a violation occurs during development, it is caught immediately rather than causing mysterious failures later.

---

### A6 — Tooling and Verification

**What it means:** Automated tools catch problems that humans miss. This policy specifies which tools run, when they run, and what must pass before code merges.

**Categories:**

- **Static analysis** examines code without running it. Tools flag potential bugs, style violations, and patterns known to cause problems.

- **Sanitizers** instrument code to detect errors at runtime. Memory sanitizers catch out-of-bounds access and use-after-free. Thread sanitizers catch data races. Undefined behavior sanitizers catch operations with unpredictable results.

- **Continuous integration** runs these tools automatically on every change. The pipeline should build and test on multiple compilers and platforms to catch portability issues.

**Practical constraint:** Not all sanitizers work equally well on all platforms. The policy should require sanitizer testing where supported, without blocking the entire strategy when a specific platform lacks coverage.

**Why it matters:** Human review cannot reliably catch memory corruption or race conditions. Automated tools can. Requiring them in CI ensures problems are caught before they reach production.

---

### A7 — Testing Patterns

**What it means:** Different kinds of tests verify different properties. This policy defines the categories every migrated module must have.

**Categories:**

- **Unit tests** verify individual components in isolation. For migrated code, this means testing the C++ implementation directly.

- **Integration tests** verify that the C interface correctly exposes the C++ implementation. These call the public C functions and confirm they behave as documented.

- **Translation tests** verify that error mapping works correctly. These deliberately trigger each error condition and confirm the C interface returns the expected code. They also verify that exceptions cannot escape the boundary.

- **Benchmark tests** measure performance. These compare the migrated implementation against the original to detect regressions.

**Why it matters:** Unit tests alone do not verify the interface layer. Integration tests alone do not isolate defects. The combination ensures both correctness and debuggability.

---

### A8 — Avoidances

**What it means:** Certain patterns cause maintenance problems that outweigh their convenience. This policy bans them.

**The bans:**

- **No raw memory management in normal code.** Direct allocation and deallocation should only appear in dedicated low-level components. All other code uses ownership wrappers that automatically manage lifetime.

- **No C-style type conversions.** Legacy C syntax for conversions does not distinguish between safe and dangerous operations. Modern C++ provides explicit conversion operators that make intent clear and searchable.

- **No premature optimization.** Performance work is guided by measurement. Changes made to "improve performance" without profiling data often make code harder to maintain without measurable benefit.

- **No monolithic service access.** If the system provides a central registry for obtaining services, split it by subsystem. A single access point that provides everything becomes a hidden dependency that couples unrelated code.

**Why it matters:** These patterns are individually convenient but collectively create code that is difficult to understand, test, and safely modify.

---

### B1 — Define C API and ABI Boundaries

**What it means:** Identify every function and data structure that external code depends on. These are the elements that must remain stable throughout and after migration.

**What to produce:**

- A list of exported functions, including their signatures and documented behavior.
- Ownership rules for every buffer or handle that crosses the boundary.
- The error model: what codes exist, what each means, and how callers retrieve additional diagnostics.

**Why first:** Everything else must fit within what this boundary can represent. If you redesign internals in a way that cannot be safely expressed through the C interface, you must either break compatibility or redo the internal design.

---

### B2 — Classify Globals and Dependencies

**What it means:** Legacy C code often uses global variables and functions that implicitly access shared state. This step inventories that state and decides how the migrated code will access it.

**Options:**

- **Immutable shared data:** Data that is set once at startup and never modified can remain globally accessible. The risk is low because no concurrent modification occurs.

- **Service registry:** A central component manages access to services that modules need. Modules request services by type; the registry provides them. This reduces the number of parameters functions must accept but makes dependencies less visible in signatures.

- **Explicit passing:** Functions receive the services they need as parameters. This makes dependencies visible and simplifies testing but increases signature size.

**What to decide for this module:** Which globals become immutable shared data, which become registry-provided services, and which become explicit parameters. The decision depends on how often the dependency is used, how stable it is, how important test isolation is, and how much signature churn the team can absorb.

**Why second:** Dependency strategy determines how much code changes when you need to swap an implementation—for testing, configuration, or future replacement. Deciding late means revisiting call sites throughout already-migrated code.

---

### B3 — Decompose into Namespace Hierarchy

**What it means:** Organize the module's code into the naming structure established in A1 before introducing new types or classes.

**What to do:** Map the legacy C files and functions to their new namespace locations. Group related functionality. Identify boundaries between areas that might become separate components.

**Why third:** This creates the structure that later changes fit into. Moving code into namespaces is low-risk and mechanical. Doing it after adding classes and inheritance relationships is much harder because those relationships constrain what can move where.

---

### B4 — Define Concurrency Requirements

**What it means:** Declare how this module behaves when accessed from multiple threads, and how it shuts down safely.

**Threading classifications:**

- **Thread-confined:** The module is only called from a single thread. No internal synchronization is needed, but callers must ensure they do not violate this constraint.

- **Thread-compatible:** Concurrent access is safe if the caller provides external synchronization. The module does not protect itself, but it does not have hidden shared state that would make external protection insufficient.

- **Thread-safe:** The module handles concurrent access internally. Callers can invoke functions from multiple threads without coordination.

**Shutdown concerns:** If the module creates background threads or holds resources that other threads might access, the shutdown sequence matters. Document who initiates shutdown, what order resources are released, and how the module ensures no access occurs after destruction begins.

**Lock ordering:** If the module uses multiple locks, document the acquisition order. Acquiring locks in inconsistent order across call sites causes deadlocks that only manifest under load. A simple rule—always acquire lock A before lock B—prevents this class of bug entirely.

**Why fourth:** Ownership and lifetime rules depend on threading. If two threads can access an object, destruction must be coordinated. If an object is confined to one thread, simpler ownership is possible. You cannot design memory management correctly without knowing the concurrency model.

---

### B5 — Decide Memory and Ownership

**What it means:** For every resource the module uses—memory, file handles, network connections, locks—define who creates it, who can access it, and who destroys it.

**Ownership patterns:**

- **Exclusive ownership:** One component is responsible for the resource's lifetime. When that component is destroyed, the resource is released. This is appropriate when the resource does not need to be shared.

- **Shared ownership:** Multiple components hold the resource, and it is released only when the last one is done. This is appropriate when true sharing is required, but it adds overhead and complexity.

- **Borrowed access:** A component uses a resource but does not control its lifetime. This is appropriate for short-term access when the owner guarantees the resource remains valid.

**What to decide:** For each resource type in the module, which pattern applies. This determines which C++ facilities (ownership wrappers, reference types, custom pools) are appropriate.

**Why fifth:** The concurrency model constrains these choices. Shared ownership across threads requires synchronization. Exclusive ownership in a thread-confined module does not. Making ownership decisions before knowing the threading model leads to either over-engineering (unnecessary synchronization) or bugs (races on destruction).

---

### B6 — Identify Compile-Time Opportunities

**What it means:** Determine which values, computations, and checks in the module can be resolved when the code is compiled rather than when it runs.

**The distinction:**

- **Compile-time** computation happens once, during the build. The result is baked into the executable. There is no runtime cost, and errors are caught before the program ever runs. This is appropriate for values that are fixed by the program's design—buffer sizes, lookup tables, configuration that varies per build but not per execution, and checks that validate type relationships or constant expressions.

- **Runtime** computation happens during execution. This is necessary when values depend on input, environment, or state that cannot be known until the program runs—user data, file contents, network responses, or hardware properties detected at startup.

**What to identify:**

- **Constants and configuration:** Are buffer sizes, table dimensions, or protocol limits fixed at build time? If so, they can be compile-time constants rather than runtime variables.

- **Validation and bounds:** Can array bounds, enum ranges, or type compatibility be verified at compile time? If so, violations become build errors rather than runtime crashes.

- **Computation:** Are there lookup tables, conversion factors, or mathematical constants that can be computed during compilation? Moving this work to compile time eliminates runtime overhead and guarantees the computation happens exactly once.

- **Type relationships:** Does the module depend on types having certain properties—specific sizes, alignments, or interface conformance? These can be verified at compile time so that incompatible types produce clear build errors rather than subtle runtime failures.

**Why now:** The previous steps established what data the module works with (B1), how it obtains dependencies (B2), and what ownership and concurrency constraints apply (B4, B5). With that context, you can now identify which of those values and constraints are genuinely dynamic versus fixed by design. This informs the next step—feature selection—because compile-time computation uses different language facilities than runtime computation.

**What to produce:** A classification of the module's significant values and checks as compile-time or runtime, with brief rationale for borderline cases. This becomes input to B7, where you select the specific language features that implement each category.

---

### B7 — Select Language Features

**What it means:** Choose the specific C++ containers, views, utilities, synchronization primitives, and compile-time facilities the module will use.

**Guidance:**

- Match containers to ownership: if data is exclusively owned and resized dynamically, a growable array is appropriate. If data is fixed at compile time, a fixed-size array avoids unnecessary overhead.

- Match views to lifetime: non-owning views of data (for read-only access to buffers or strings) are efficient but dangerous if the underlying data is freed while the view exists. Only use views when lifetime is explicitly managed and documented.

- Match synchronization to the concurrency model: if the module is thread-confined, no locks are needed. If shared state exists, choose primitives that match the access patterns—locks for infrequent access, lock-free structures only when profiling justifies the complexity.

- Match compile-time facilities to the classification from B6: values identified as compile-time constants use constant expressions; compile-time checks use static assertions; compile-time computation uses facilities that evaluate during the build.

**Why last among design steps:** Features must serve the design, not drive it. Choosing a container before understanding ownership leads to fighting the container's semantics. Choosing synchronization primitives before understanding threading leads to either deadlocks or unnecessary contention. Choosing runtime facilities for compile-time-known values wastes performance and moves error detection later than necessary.

---

### B8 — Write Tests

**What it means:** Implement the test categories defined in A7 for this specific module.

**What to cover:**

- The C++ implementation in isolation (unit tests).
- The C interface with both success and failure paths (integration tests).
- Every error code the interface can return, verifying correct mapping (translation tests).
- A deliberate exception-triggering path, verifying it does not escape the C boundary.

**Why now:** Testing after the design is complete but before declaring the module finished ensures the design is validated. Testing earlier risks testing code that will change; testing later risks shipping unvalidated code.

---

### B9 — Validate Performance

**What it means:** Measure the migrated implementation against the original to detect regressions.

**Approach:**

- Establish a baseline by measuring the original C implementation under a representative workload.
- Measure the migrated implementation under the same workload.
- Define an acceptable threshold—for example, no more than five percent slower on critical paths.
- If the threshold is exceeded, profile to identify the cause before deciding whether to accept, optimize, or redesign.

**Nuance:** Microbenchmarks can mislead. A function that is five percent slower in isolation may have no measurable impact on real workloads. Conversely, a function that appears fine in microbenchmarks may cause problems at scale. Use representative workloads when possible, and document assumptions when only microbenchmarks are available.

**Why now:** Performance problems are cheaper to fix before the module is integrated and other code depends on its current design. Measuring at the end of each module prevents accumulating regressions that only become visible much later.

---

### B10 — Update Documentation

**What it means:** Record the decisions made during migration so future work does not contradict them.

**What to record:**

- Boundary summary: what is exported, what stability guarantees apply.
- Dependency decisions: what global state exists, how services are accessed.
- Concurrency and ownership: threading classification, shutdown ordering, resource lifetimes.
- Error handling: which functions use expected values, which throw, how the boundary translates them.
- Test coverage: what is tested, what thresholds apply.
- Known limitations: what is intentionally deferred, what risks remain.

**Why last:** Documentation written before decisions are final becomes outdated immediately. Documentation written after the module is complete captures the actual state. Keeping it as the final step ensures accuracy.
