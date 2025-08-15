"""
Basic test script for function calling capabilities without requiring OpenAI API.

This script tests the driving behavior analysis functions locally to ensure they work correctly
before integrating with GPT function calling.
"""

import json
import sys
from typing import Dict, List, Any
from test_gpt_function_calling import DrivingBehaviorFunctions

class BasicFunctionTester:
    """
    Test the driving behavior functions locally without GPT integration.
    """
    
    def __init__(self):
        self.functions = DrivingBehaviorFunctions()
    
    def test_red_light_scenarios(self) -> List[Dict[str, Any]]:
        """Test various red light scenarios."""
        scenarios = [
            {
                "name": "Clear Red Light Violation",
                "params": {
                    "vehicle_position": "approaching",
                    "traffic_light_status": "red", 
                    "intersection_crossed": True,
                    "timestamp": "2024-02-06T14:30:00"
                },
                "expected_violation": True
            },
            {
                "name": "No Violation - Stopped at Red",
                "params": {
                    "vehicle_position": "at_line",
                    "traffic_light_status": "red",
                    "intersection_crossed": False, 
                    "timestamp": "2024-02-06T14:31:00"
                },
                "expected_violation": False
            },
            {
                "name": "No Violation - Green Light",
                "params": {
                    "vehicle_position": "entering",
                    "traffic_light_status": "green",
                    "intersection_crossed": True,
                    "timestamp": "2024-02-06T14:32:00"
                },
                "expected_violation": False
            }
        ]
        
        results = []
        for scenario in scenarios:
            result = self.functions.analyze_red_light_violation(**scenario["params"])
            test_passed = result["violation_detected"] == scenario["expected_violation"]
            
            results.append({
                "scenario_name": scenario["name"],
                "test_passed": test_passed,
                "expected": scenario["expected_violation"],
                "actual": result["violation_detected"], 
                "full_result": result
            })
        
        return results
    
    def test_stop_sign_scenarios(self) -> List[Dict[str, Any]]:
        """Test various stop sign scenarios."""
        scenarios = [
            {
                "name": "Rolling Stop Violation",
                "params": {
                    "vehicle_speed": 8.5,
                    "stop_sign_present": True,
                    "complete_stop_made": False,
                    "location": "Main St & 1st Ave"
                },
                "expected_violation": True
            },
            {
                "name": "Complete Stop - No Violation", 
                "params": {
                    "vehicle_speed": 0.0,
                    "stop_sign_present": True,
                    "complete_stop_made": True,
                    "location": "Oak St & 2nd Ave"
                },
                "expected_violation": False
            },
            {
                "name": "No Stop Sign Present",
                "params": {
                    "vehicle_speed": 15.0,
                    "stop_sign_present": False,
                    "complete_stop_made": False,
                    "location": "Highway 101"
                },
                "expected_violation": False
            }
        ]
        
        results = []
        for scenario in scenarios:
            result = self.functions.analyze_stop_sign_violation(**scenario["params"])
            test_passed = result["violation_detected"] == scenario["expected_violation"]
            
            results.append({
                "scenario_name": scenario["name"],
                "test_passed": test_passed,
                "expected": scenario["expected_violation"],
                "actual": result["violation_detected"],
                "full_result": result
            })
        
        return results
    
    def test_regulation_lookup(self) -> List[Dict[str, Any]]:
        """Test traffic regulation lookup functionality."""
        test_cases = [
            {
                "violation_type": "red_light",
                "state": "Washington",
                "expected_statute": "RCW 46.61.055"
            },
            {
                "violation_type": "stop_sign", 
                "state": "Washington",
                "expected_statute": "RCW 46.61.065"
            },
            {
                "violation_type": "unknown_violation",
                "state": "Washington", 
                "expected_statute": "Unknown"
            }
        ]
        
        results = []
        for case in test_cases:
            result = self.functions.get_traffic_regulation(
                case["violation_type"], 
                case["state"]
            )
            test_passed = result["statute"] == case["expected_statute"]
            
            results.append({
                "test_case": f"{case['violation_type']} in {case['state']}",
                "test_passed": test_passed,
                "expected_statute": case["expected_statute"],
                "actual_statute": result["statute"],
                "full_result": result
            })
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all basic function tests."""
        print("🧪 Running Basic Function Tests")
        print("=" * 50)
        
        # Test red light scenarios
        print("\n🚦 Testing Red Light Analysis Functions...")
        red_light_results = self.test_red_light_scenarios()
        red_light_passed = sum(1 for r in red_light_results if r["test_passed"])
        
        print(f"Red Light Tests: {red_light_passed}/{len(red_light_results)} passed")
        for result in red_light_results:
            status = "✓" if result["test_passed"] else "✗"
            print(f"  {status} {result['scenario_name']}")
        
        # Test stop sign scenarios
        print("\n🛑 Testing Stop Sign Analysis Functions...")
        stop_sign_results = self.test_stop_sign_scenarios()
        stop_sign_passed = sum(1 for r in stop_sign_results if r["test_passed"])
        
        print(f"Stop Sign Tests: {stop_sign_passed}/{len(stop_sign_results)} passed")
        for result in stop_sign_results:
            status = "✓" if result["test_passed"] else "✗"
            print(f"  {status} {result['scenario_name']}")
        
        # Test regulation lookup
        print("\n📋 Testing Regulation Lookup Functions...")
        regulation_results = self.test_regulation_lookup()
        regulation_passed = sum(1 for r in regulation_results if r["test_passed"])
        
        print(f"Regulation Tests: {regulation_passed}/{len(regulation_results)} passed")
        for result in regulation_results:
            status = "✓" if result["test_passed"] else "✗"
            print(f"  {status} {result['test_case']}")
        
        # Overall summary
        total_tests = len(red_light_results) + len(stop_sign_results) + len(regulation_results)
        total_passed = red_light_passed + stop_sign_passed + regulation_passed
        
        print(f"\n📊 Overall Results:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Success Rate: {(total_passed/total_tests)*100:.1f}%")
        
        return {
            "red_light_results": red_light_results,
            "stop_sign_results": stop_sign_results, 
            "regulation_results": regulation_results,
            "summary": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "success_rate": (total_passed/total_tests)*100
            }
        }

def main():
    """Main function to run basic tests."""
    tester = BasicFunctionTester()
    results = tester.run_all_tests()
    
    # Return non-zero exit code if any tests failed
    if results["summary"]["total_passed"] < results["summary"]["total_tests"]:
        print("\n❌ Some tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()