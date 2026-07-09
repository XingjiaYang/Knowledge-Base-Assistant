from pathlib import Path
import sys
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.call_graph import _call_name_candidates, _qualified_suffixes, extract_call_edges
from app.code_indexer import CodeFileRecord, parse_code_functions


def _record(path: str, language: str, text: str) -> CodeFileRecord:
    return CodeFileRecord(
        id=uuid4(),
        repository_id="code",
        repository_name="code",
        path=path,
        source_root="data/code",
        language=language,
        full_content=text,
        content_sha256="test",
        line_count=len(text.splitlines()),
    )


def assert_python_symbols_and_calls() -> None:
    source = '''class DMatrix:
    def __init__(self, data):
        """load data"""
        self.data = data
        helper(data)

def helper(x):
    return x
'''
    functions = parse_code_functions(_record("xgboost/core.py", "python", source), source)
    symbols = [(item.kind, item.qualified_name) for item in functions]
    if ("class", "DMatrix") not in symbols:
        raise AssertionError(f"Python class not extracted: {symbols}")
    if ("function", "DMatrix.__init__") not in symbols:
        raise AssertionError(f"Python method not extracted: {symbols}")

    edges = {(edge.caller_name, edge.callee_name) for edge in extract_call_edges(functions)}
    if ("DMatrix.__init__", "helper") not in edges:
        raise AssertionError(f"Python call edge not extracted: {edges}")
    print("Python code AST extraction -> ok")


def assert_cpp_symbols_and_calls() -> None:
    source = """namespace xgboost {
class Foo {
 public:
  void Run() { Bar(); }
};
void Bar() {}
void Configure(Context const& ctx) { Use(ctx); }
}
"""
    functions = parse_code_functions(_record("src/foo.cc", "cpp", source), source)
    symbols = [(item.kind, item.qualified_name) for item in functions]
    if ("class", "Foo") not in symbols:
        raise AssertionError(f"C++ class not extracted: {symbols}")
    if ("function", "Foo::Run") not in symbols:
        raise AssertionError(f"C++ method not extracted: {symbols}")
    if ("function", "Configure") not in symbols:
        raise AssertionError(f"C++ free function not extracted: {symbols}")
    if any(symbol[1] == "ctx" for symbol in symbols):
        raise AssertionError(f"C++ parameter was extracted as function: {symbols}")

    edges = {(edge.caller_name, edge.callee_name) for edge in extract_call_edges(functions)}
    if ("Foo::Run", "Bar") not in edges:
        raise AssertionError(f"C++ call edge not extracted: {edges}")
    print("C++ code AST extraction -> ok")


def assert_call_name_resolution_helpers() -> None:
    python_candidates = _call_name_candidates("self.transform", "DMatrix.__init__")
    if "DMatrix.transform" not in python_candidates:
        raise AssertionError(f"Python self-call candidate missing: {python_candidates}")

    cpp_candidates = _call_name_candidates("this->Run", "LearnerImpl::Configure")
    if "LearnerImpl::Run" not in cpp_candidates:
        raise AssertionError(f"C++ this-call candidate missing: {cpp_candidates}")

    suffixes = _qualified_suffixes("xgboost::common::AssertGPUSupport")
    if "common::AssertGPUSupport" not in suffixes or "AssertGPUSupport" not in suffixes:
        raise AssertionError(f"Qualified suffixes missing: {suffixes}")
    print("Call graph name resolution helpers -> ok")


def main() -> None:
    assert_python_symbols_and_calls()
    assert_cpp_symbols_and_calls()
    assert_call_name_resolution_helpers()


if __name__ == "__main__":
    main()
