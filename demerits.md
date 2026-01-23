# Demerits

| Date | Issue | Description | Status |
|------|-------|-------------|--------|
| 2026-01-23 | Workaround instead of fix | Encountered missing `REQUIRE_THROWS` macro in test framework. Instead of adding the macro, used `REQUIRE_THROWS_AS(expr, std::runtime_error)` as workaround. | **FIXED** - Added proper `REQUIRE_THROWS` macro |
| 2026-01-23 | Document instead of fix | P1 #5 Session/Graph lock coordination - chose to "document the limitation" instead of implementing proper lock coordination between Session::Run() and Graph mutation. | **FIXED** - Implemented freeze() approach: Graph becomes immutable after Session creation |
| 2026-01-23 | Inconsistent contract | Graph moved-from semantics: Tests said "valid empty" but implementation threw on use. ChatGPT caught this inconsistency. | **FIXED** - Chose "throw on use" contract, updated all tests to match |
| 2026-01-23 | CMake export bug | `tf::wrapper` was only a build-tree ALIAS, not available after `find_package()`. | **FIXED** - Added `EXPORT_NAME wrapper` so exported target is `tf::wrapper` |
| 2026-01-23 | `FromVector<bool>` broken | `std::vector<bool>::data()` doesn't exist because `vector<bool>` is a bitfield. Code would not compile. Tests worked around it by using int32_t and casting. | **FIXED** - Used `if constexpr` to copy element-by-element for bool type |
| 2026-01-23 | Non-existent `Zeros` op | `Zeros` op doesn't exist in TensorFlow C API. Would cause runtime error. Tests skipped it with a comment. | **FIXED** - Removed op, added documentation pointing to `Fill` or `ZerosLike` |
| 2026-01-23 | Non-existent `Dropout` op | `Dropout` op doesn't exist in TensorFlow C API. Would cause runtime error. Tests skipped it with a comment. | **FIXED** - Removed op, added documentation on implementing dropout with primitives |
