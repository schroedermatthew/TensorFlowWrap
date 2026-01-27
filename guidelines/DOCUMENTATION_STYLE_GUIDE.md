# TensorFlowWrap Documentation Style Guide

**Authority:** Subordinate to the *TensorFlowWrap Development Guidelines*  
**Version:** 1.0 (January 2026)

---

## Purpose

This guide defines standards for TensorFlowWrap documentation. Good documentation:

- Helps users integrate TensorFlowWrap into their applications
- Explains design decisions and tradeoffs
- Provides concrete, tested examples
- Avoids marketing language in favor of precise technical claims

---

## Document Types

| Doc Type | Question It Answers | When to Use |
|----------|---------------------|-------------|
| **README** | "What is this and how do I start?" | Repository entry point |
| **Design Document** | "Why is it designed this way?" | Architecture, rationale |
| **API Reference** | "What does this function do?" | In-code documentation |
| **User Guide** | "How do I accomplish X?" | Task-oriented tutorials |
| **Style Guide** | "How should I write X?" | Standards documents |

---

## File Naming

| Document Type | Format | Example |
|---------------|--------|---------|
| Design docs | `SCREAMING_SNAKE.md` | `DESIGN.md` |
| Style guides | `SCREAMING_SNAKE.md` | `TEST_STYLE_GUIDE.md` |
| User guides | `Title Case.md` | `Getting Started.md` |
| API docs | In-code comments | N/A |

---

## Writing Standards

### Voice and Tone

- **Active voice**: "TensorFlowWrap wraps TensorFlow resources" not "Resources are wrapped"
- **Direct**: "Use `FromScalar` to create a scalar tensor" not "One might consider using..."
- **Technical precision**: Avoid vague adjectives

### Banned Vocabulary

These words are banned in documentation because they're vague or misleading:

| Banned | Use Instead |
|--------|-------------|
| "fast" | "O(1) lookup", "zero-copy", specific benchmark numbers |
| "efficient" | Specific memory/time complexity |
| "simple" | "N lines of code", "single function call" |
| "easy" | Describe the actual steps |
| "lightweight" | Header-only, no runtime dependencies, specific binary size |
| "powerful" | List specific capabilities |
| "flexible" | Describe actual extension points |
| "modern" | "C++20", specific features used |
| "robust" | Describe error handling, edge case behavior |
| "seamless" | Describe the actual integration steps |

**Example:**

```markdown
# Bad
TensorFlowWrap provides a fast, efficient, easy-to-use wrapper.

# Good
TensorFlowWrap provides RAII wrappers with zero overhead over direct C API 
calls. Loading a SavedModel requires one function call. All TensorFlow 
resources are automatically released when wrappers go out of scope.
```

### Code Examples

All code examples must be:
1. **Complete**: Can be copied and compiled
2. **Tested**: Actually verified to work
3. **Minimal**: Only includes relevant code

```cpp
// Good: Complete, minimal example
#include "tf_wrap/core.hpp"

int main() {
    auto model = tf_wrap::Model::Load("model_path");
    auto input = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto result = model.runner()
        .feed(model.input("x"), input)
        .fetch(model.output("y"))
        .run_one();
    
    for (float v : result.read<float>().span()) {
        std::cout << v << "\n";
    }
}
```

```cpp
// Bad: Incomplete snippet
auto result = model.Run(...);  // What is model? What are the args?
```

### Prose Over Lists and Tables

Prefer natural prose over bullet lists and tables. Lists and tables are appropriate only when:
- Comparing multiple items across the same dimensions
- Presenting reference data (API signatures, error codes)
- The structure genuinely aids comprehension

```markdown
# Good: Prose
Create tensors with `FromScalar<T>(value)` for single values, 
`FromVector<T>(shape, data)` for arrays, or `Zeros<T>(shape)` for 
zero-initialized tensors. Use `Allocate<T>(shape)` when you need to 
write data directly.

# Bad: Unnecessary list
Ways to create tensors:
- `FromScalar<T>(value)` - for single values
- `FromVector<T>(shape, data)` - for arrays  
- `Zeros<T>(shape)` - for zero-initialized
- `Allocate<T>(shape)` - for writing data
```

Tables are acceptable for reference material:

```markdown
| Method | Returns | Throws |
|--------|---------|--------|
| `FromScalar<T>` | `Tensor` | `Error` on allocation failure |
| `read<T>()` | `TensorView<const T>` | `Error` on dtype mismatch |
```

---

## API Documentation

### Function Documentation Format

```cpp
/// Brief one-line description.
///
/// Longer description if needed. Explain behavior, not implementation.
///
/// @tparam T Template parameter description
/// @param name Parameter description
/// @return What is returned
/// @throws ExceptionType When this is thrown
/// @note Important usage notes
/// @see Related functions
///
/// Example:
/// @code
/// auto t = Tensor::FromScalar<float>(3.14f);
/// @endcode
template<TensorScalar T>
[[nodiscard]] static Tensor FromScalar(T value);
```

### What to Document

| Element | Required Documentation |
|---------|------------------------|
| Public class | Purpose, thread safety, ownership |
| Public method | Brief, params, return, throws |
| Template parameters | Constraints, valid types |
| Factory functions | What they create, when to use |
| Error conditions | When exceptions are thrown |

### What Not to Document

- Private implementation details
- Obvious getters (`handle()` returns the handle)
- Self-documenting code

---

## Design Documents

### Required Sections

Every design document must include:

1. **Executive Summary**: What and why in 2-3 sentences
2. **Problem Statement**: What problem does this solve
3. **Solution**: How it solves the problem
4. **Design Decisions**: Key choices and rationale
5. **Tradeoffs**: What was sacrificed for what

### Design Decision Format

```markdown
### Why X Instead of Y?

**Decision:** We chose X.

**Alternatives considered:**
1. Y - Rejected because [reason]
2. Z - Rejected because [reason]

**Rationale:** X provides [benefit] while Y would require [cost].
```

---

## README Structure

The main README should follow this structure:

```markdown
# TensorFlowWrap

Brief description (1-2 sentences).

## Features

- Feature 1
- Feature 2

## Quick Start

Minimal working example.

## Requirements

- C++20 compiler
- TensorFlow C library 2.13+

## Installation

Step-by-step instructions.

## Documentation

Links to other docs.

## License

License statement.
```

---

## Markdown Standards

### Headers

- Use ATX-style headers (`#`, `##`, etc.)
- One H1 (`#`) per document
- Don't skip levels (no H1 → H3)

### Code Blocks

- Always specify language: ` ```cpp `
- Use 4-space indentation in code
- Keep examples under 30 lines when possible

### Links

- Use reference-style links for repeated URLs
- Prefer relative links within the repository

```markdown
See the [Design Document](DESIGN.md) for architecture details.
```

### Tables

- Align columns with spaces
- Use `|` separators
- Include header row

---

## Checklist

Before committing documentation:

- [ ] No banned vocabulary
- [ ] Code examples compile and run
- [ ] Prose used instead of unnecessary lists/tables
- [ ] Active voice throughout
- [ ] Technical claims are specific and verifiable
- [ ] Links work
- [ ] Markdown renders correctly

---

## Examples

### Good Documentation

```markdown
## Tensor Creation

Create tensors using factory methods. Use `FromScalar<T>(value)` for single 
values, `FromVector<T>(shape, data)` for array data, `Zeros<T>(shape)` for 
zero-initialized tensors, or `Allocate<T>(shape)` for uninitialized tensors 
that you'll write to directly.

```cpp
// Create a 2x3 float tensor
auto t = tf_wrap::Tensor::FromVector<float>({2, 3}, {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
});
```
```

### Bad Documentation

```markdown
## Tensor Creation

TensorFlowWrap provides a powerful and flexible tensor creation API that 
makes it easy to create tensors in a simple and efficient way. The modern 
C++20 interface is robust and seamless to use.

You can use FromScalar to create scalars, FromVector to create vectors, 
Zeros to create zeros, and Allocate to allocate memory.
```

---

*TensorFlowWrap Documentation Style Guide v1.0 -- January 2026*
