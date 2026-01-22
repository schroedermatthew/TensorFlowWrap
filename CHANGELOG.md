# CHANGELOG - TensorFlowWrap Bug Fixes

## v4.1 - Bug Fixes (2026-01-22)

### Critical Bug Fixes

#### H1: Clone() Race Condition - FIXED
**File:** `include/tf_wrap/tensor.hpp` (lines 448-476)

**Problem:** `Clone()` read tensor data without acquiring any lock. When using 
`SharedTensor` or `SafeTensor`, if another thread was writing via `write<T>()`,
the clone operation could read partially-written (torn) data, causing silent 
data corruption.

**Root Cause:** The lambda in `Clone()` captured `this` and directly accessed
`TF_TensorData(state_->tensor)` without synchronization.

**Fix:** Added `scoped_shared()` lock before reading data. Changed lambda to 
capture the data pointer (`src`) instead of `this`, ensuring the copy happens
entirely under lock protection.

**Test Added:** `test_comprehensive_bugs.cpp` - "H1-BUG: Clone during concurrent 
write detects torn reads"

---

#### H2: DebugString() Deadlock - FIXED
**File:** `include/tf_wrap/graph.hpp` (lines 612-640)

**Problem:** `SafeGraph::DebugString()` would deadlock immediately when called.
The method acquired a lock via `scoped_shared()`, then called `num_operations()`
which attempted to acquire the same lock. Since `std::mutex` is non-recursive,
this caused an immediate deadlock.

**Root Cause:** Nested lock acquisition in methods that call other locking methods.

**Fix:** Inlined the operation counting logic directly in `DebugString()` instead
of calling `num_operations()`. The method now counts operations in a single pass
while holding the lock, then uses that count.

**Test Added:** `test_comprehensive_bugs.cpp` - "H2-BUG: SafeGraph DebugString 
deadlock check" (uses async with timeout to detect deadlock)

---

### Test Suite Additions

**New File:** `tests/test_comprehensive_bugs.cpp`

Added 17 new tests covering:
- Clone() thread safety (stress test with concurrent writer + cloners)
- SafeGraph DebugString deadlock detection
- SafeGraph full API coverage
- SafeTensor / SharedTensor concurrent access
- View lifetime safety (view outlives tensor)
- Error consistency checks

**Run with:**
```bash
# Basic tests
./test_comprehensive_bugs

# Include stress tests
./test_comprehensive_bugs --stress

# Include bug demonstration tests
./test_comprehensive_bugs --bugs

# Run everything
./test_comprehensive_bugs --all
```

---

### Why CI Previously Passed

The original CI only tested:
- `FastGraph` (NoLock policy) - never `SafeGraph` (Mutex policy)
- `FastTensor` (NoLock policy) - never stressed `SharedTensor`/`SafeTensor` Clone()
- Single-threaded Clone operations

Both bugs only manifested with the thread-safe variants (`Safe*` and `Shared*`),
which were not exercised in the original test suite.

---

### Verification

After applying fixes:
- All 36 original tests pass ✓
- All 31 edge case tests pass ✓  
- All 17 new comprehensive bug tests pass ✓
- ThreadSanitizer reports no races ✓
