import pytest

### PyTest Benchmark Hooks ###
@pytest.mark.hookwrapper
def pytest_benchmark_generate_json(config, benchmarks, include_data, machine_info, commit_info):
    """Disables pytest-benchmark from stripping benchmark results from errored tests.
    See: https://pytest-benchmark.readthedocs.io/en/latest/hooks.html.
    """
    for bench in benchmarks:
        if bench.has_error or len(bench.stats.data) == 0:
            bench.stats.data = [float("nan")]
        bench.extra_info["has_error"] = bench.has_error  # record errored state here
        bench.fixture.has_error = False  # always set to false so it always outputs in report
    yield

# def pytest_benchmark_update_json(config, benchmarks, output_json):
#     output_json["foo"] = "bar"