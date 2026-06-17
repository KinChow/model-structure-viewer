from model_structure_viewer.structure.repair import IntrospectionFailureKind, RepairContext, RepairResult
from model_structure_viewer.structure.repair.runner import try_repair


class MatchingStrategy:
    name = "matching"

    def matches(self, context):
        return True

    def apply(self, context):
        return RepairResult(
            config={**context.config, "patched": True},
            local_dir=context.local_dir,
            strategy_name=self.name,
            diagnostics={"repair_status": "success"},
        )


class NonMatchingStrategy:
    name = "non_matching"

    def matches(self, context):
        return False

    def apply(self, context):
        raise AssertionError("should not be called")


class FailingStrategy:
    name = "failing"

    def matches(self, context):
        return True

    def apply(self, context):
        raise RuntimeError("repair failed")


def _context():
    return RepairContext(
        config={"model_type": "demo"},
        source={},
        local_dir=None,
        failure_kind=IntrospectionFailureKind.UNKNOWN,
        original_error="failed",
    )


def test_try_repair_returns_result_for_matching_strategy():
    result = try_repair(_context(), strategies=[MatchingStrategy()])

    assert result is not None
    assert result.config["patched"] is True
    assert result.strategy_name == "matching"


def test_try_repair_returns_none_without_matching_strategy():
    assert try_repair(_context(), strategies=[NonMatchingStrategy()]) is None


def test_try_repair_converts_strategy_exception_to_none():
    assert try_repair(_context(), strategies=[FailingStrategy()]) is None
