# TensorFlowWrap Code Coverage

Generate code coverage reports using gcov and lcov.

## Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install lcov

# macOS
brew install lcov
```

## Generating Coverage

```bash
# Run from repo root
./tests/coverage/coverage.sh

# Custom output directory
./tests/coverage/coverage.sh /path/to/output
```

## Output

- `coverage/index.html` - HTML report (open in browser)
- `build/coverage/coverage_final.info` - lcov data file

## CI Integration

Coverage is generated on every push and uploaded as an artifact:

```yaml
- name: Generate coverage
  run: ./tests/coverage/coverage.sh

- name: Upload coverage
  uses: actions/upload-artifact@v4
  with:
    name: coverage-report
    path: coverage/
```

## Interpreting Results

| Metric | Target | Description |
|--------|--------|-------------|
| Line coverage | >80% | Percentage of executable lines hit |
| Function coverage | >90% | Percentage of functions called |
| Branch coverage | >70% | Percentage of branches taken |

## Coverage by Component

The report breaks down coverage by header file:

- `tf_wrap/tensor.hpp` - Tensor operations
- `tf_wrap/session.hpp` - Session management
- `tf_wrap/graph.hpp` - Graph operations
- `tf_wrap/facade.hpp` - High-level Model/Runner API
- `tf_wrap/small_vector.hpp` - SmallVector container
- `tf_wrap/scope_guard.hpp` - RAII scope guard
- `tf_wrap/status.hpp` - TensorFlow status wrapper
- `tf_wrap/error.hpp` - Error handling
- `tf_wrap/format.hpp` - String formatting

## Improving Coverage

1. Identify uncovered lines in the HTML report
2. Add tests targeting those code paths
3. Re-run coverage to verify improvement

Common gaps:
- Error handling branches
- Edge cases (empty inputs, max sizes)
- Rarely-used API overloads
