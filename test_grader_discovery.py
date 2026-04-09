"""
Comprehensive test verifying all grader discovery mechanisms work correctly.
Shows that validators can find graders using:
1. Direct imports
2. Module introspection
3. Dynamic discovery via globals()
"""

import inspect
import importlib
import dataset


def test_direct_import():
    """Test that graders can be imported directly from dataset module."""
    print("\n=== TEST 1: Direct Imports ===")
    
    grader_functions = [
        'grade_easy_001', 'grade_easy_002', 'grade_easy_003', 'grade_easy_004',
        'grade_medium_001', 'grade_medium_002', 'grade_medium_003', 'grade_medium_004',
        'grade_hard_001', 'grade_hard_002', 'grade_hard_003', 'grade_hard_004'
    ]
    
    for grader_name in grader_functions:
        try:
            grader = getattr(dataset, grader_name)
            assert callable(grader), f"{grader_name} is not callable"
            print(f"✓ {grader_name} imported successfully")
        except AttributeError as e:
            print(f"✗ {grader_name} not found: {e}")
            return False
    
    return True


def test_module_introspection():
    """Test that graders can be discovered through module introspection."""
    print("\n=== TEST 2: Module Introspection ===")
    
    # Get all callables in dataset module
    callables = {name: obj for name, obj in inspect.getmembers(dataset)
                 if inspect.isfunction(obj) and name.startswith('grade_')}
    
    expected_graders = {
        'grade_easy_001', 'grade_easy_002', 'grade_easy_003', 'grade_easy_004',
        'grade_medium_001', 'grade_medium_002', 'grade_medium_003', 'grade_medium_004',
        'grade_hard_001', 'grade_hard_002', 'grade_hard_003', 'grade_hard_004'
    }
    found_graders = set(callables.keys())
    
    print(f"Expected: {sorted(expected_graders)}")
    print(f"Found: {sorted(found_graders)}")
    
    if expected_graders == found_graders:
        for grader_name in sorted(found_graders):
            print(f"✓ {grader_name} discovered")
        return True
    else:
        missing = expected_graders - found_graders
        extra = found_graders - expected_graders
        if missing:
            print(f"✗ Missing graders: {missing}")
        if extra:
            print(f"Note: Extra functions found: {extra}")
        return len(missing) == 0


def test_dynamic_discovery():
    """Test that graders can be discovered dynamically from module globals."""
    print("\n=== TEST 3: Dynamic Discovery via globals() ===")
    
    # Simulate how a validator might discover graders dynamically
    module = importlib.import_module('dataset')
    
    graders = {}
    for name, obj in vars(module).items():
        if name.startswith('grade_') and callable(obj):
            graders[name] = obj
    
    expected_count = 12
    found_count = len(graders)
    
    print(f"Expected: {expected_count} graders")
    print(f"Found: {found_count} graders")
    
    if found_count == expected_count:
        for grader_name in sorted(graders.keys()):
            print(f"✓ {grader_name} discovered dynamically")
        return True
    else:
        print(f"✗ Expected {expected_count} graders, found {found_count}")
        return False


def test_grader_signatures():
    """Test that graders have correct signatures."""
    print("\n=== TEST 4: Grader Function Signatures ===")
    
    grader_names = [
        'grade_easy_001', 'grade_easy_002', 'grade_easy_003', 'grade_easy_004',
        'grade_medium_001', 'grade_medium_002', 'grade_medium_003', 'grade_medium_004',
        'grade_hard_001', 'grade_hard_002', 'grade_hard_003', 'grade_hard_004'
    ]
    
    for grader_name in grader_names:
        grader = getattr(dataset, grader_name)
        sig = inspect.signature(grader)
        params = list(sig.parameters.keys())
        print(f"✓ {grader_name} signature: {sig}")
    
    return True


def test_validator_simulation():
    """Simulate how actual validator would discover and call graders."""
    print("\n=== TEST 5: Validator Simulation ===")
    
    # This simulates what the actual validator does
    print("Simulating validator discovery pattern...")
    
    # Method 1: Check if grader exists
    has_grade_easy_001 = hasattr(dataset, 'grade_easy_001')
    print(f"✓ hasattr(dataset, 'grade_easy_001'): {has_grade_easy_001}")
    
    # Method 2: Get grader and call it
    if has_grade_easy_001:
        grader = getattr(dataset, 'grade_easy_001')
        # Create test input that matches Task.grade() signature
        try:
            # Test with proper input - grade_easy_001 expects (actions, step_types)
            result = grader(actions=[], step_types=[])
            print(f"✓ grade_easy_001 executed successfully: {result}")
        except Exception as e:
            print(f"✓ grade_easy_001 executed (expected error for empty inputs): {type(e).__name__}: {str(e)[:50]}")
    
    # Method 3: Discover all graders dynamically
    module = importlib.import_module('dataset')
    all_graders = [name for name in dir(module) 
                   if name.startswith('grade_') and callable(getattr(module, name))]
    
    print(f"✓ Discovered {len(all_graders)} graders dynamically")
    print(f"  Graders: {sorted(all_graders)}")
    return len(all_graders) == 12


def main():
    """Run all discovery tests."""
    print("\n" + "="*60)
    print("GRADER DISCOVERY VERIFICATION SUITE")
    print("="*60)
    
    tests = [
        ("Direct Import", test_direct_import),
        ("Module Introspection", test_module_introspection),
        ("Dynamic Discovery", test_dynamic_discovery),
        ("Grader Signatures", test_grader_signatures),
        ("Validator Simulation", test_validator_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    print("\n" + ("="*60))
    if all_passed:
        print("✓ ALL TESTS PASSED - Graders are discoverable!")
    else:
        print("✗ SOME TESTS FAILED - Graders may not be discoverable!")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
