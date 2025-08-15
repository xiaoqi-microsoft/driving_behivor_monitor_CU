"""
Test module for GPT function calling capabilities in driving behavior monitoring.

This module tests GPT's function calling abilities for analyzing driving behavior
violations, particularly red light and stop sign infractions.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class DrivingBehaviorFunctions:
    """
    Collection of functions that can be called by GPT to analyze driving behavior.
    These functions simulate the analysis of driving violations.
    """
    
    @staticmethod
    def analyze_red_light_violation(
        vehicle_position: str,
        traffic_light_status: str,
        intersection_crossed: bool,
        timestamp: str
    ) -> Dict[str, Any]:
        """
        Analyze if a red light violation occurred.
        
        Args:
            vehicle_position: Position of vehicle relative to intersection
            traffic_light_status: Current status of traffic light (red/yellow/green)
            intersection_crossed: Whether vehicle crossed the intersection
            timestamp: When the event occurred
            
        Returns:
            Dict containing violation analysis results
        """
        violation_detected = (
            traffic_light_status.lower() == "red" and 
            intersection_crossed and 
            vehicle_position.lower() in ["approaching", "at_line", "entering"]
        )
        
        return {
            "violation_type": "red_light_violation",
            "violation_detected": violation_detected,
            "severity": "high" if violation_detected else "none",
            "details": {
                "vehicle_position": vehicle_position,
                "traffic_light_status": traffic_light_status,
                "intersection_crossed": intersection_crossed,
                "timestamp": timestamp
            },
            "recommendation": "Issue citation" if violation_detected else "No action needed"
        }
    
    @staticmethod
    def analyze_stop_sign_violation(
        vehicle_speed: float,
        stop_sign_present: bool,
        complete_stop_made: bool,
        location: str
    ) -> Dict[str, Any]:
        """
        Analyze if a stop sign violation occurred.
        
        Args:
            vehicle_speed: Speed of vehicle at stop sign (mph)
            stop_sign_present: Whether a stop sign is present
            complete_stop_made: Whether vehicle came to complete stop
            location: Location of the stop sign
            
        Returns:
            Dict containing violation analysis results
        """
        violation_detected = (
            stop_sign_present and 
            not complete_stop_made and 
            vehicle_speed > 5.0  # Rolling stop threshold
        )
        
        return {
            "violation_type": "stop_sign_violation",
            "violation_detected": violation_detected,
            "severity": "medium" if violation_detected else "none",
            "details": {
                "vehicle_speed": vehicle_speed,
                "stop_sign_present": stop_sign_present,
                "complete_stop_made": complete_stop_made,
                "location": location
            },
            "recommendation": "Issue warning" if violation_detected else "No action needed"
        }
    
    @staticmethod
    def get_traffic_regulation(violation_type: str, state: str = "Washington") -> Dict[str, Any]:
        """
        Get traffic regulation information for a specific violation type.
        
        Args:
            violation_type: Type of violation (red_light, stop_sign, etc.)
            state: State for which to get regulations
            
        Returns:
            Dict containing regulation information
        """
        regulations = {
            "red_light": {
                "Washington": {
                    "statute": "RCW 46.61.055",
                    "description": "Vehicle operators must stop at red traffic signals",
                    "penalty": "Traffic infraction with fine",
                    "fine_range": "$136-$341"
                }
            },
            "stop_sign": {
                "Washington": {
                    "statute": "RCW 46.61.065", 
                    "description": "Vehicle operators must come to complete stop at stop signs",
                    "penalty": "Traffic infraction with fine",
                    "fine_range": "$136-$207"
                }
            }
        }
        
        return regulations.get(violation_type, {}).get(state, {
            "statute": "Unknown",
            "description": "Regulation not found",
            "penalty": "Unknown",
            "fine_range": "Unknown"
        })

class GPTFunctionCallingTester:
    """
    Main test class for GPT function calling with driving behavior analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the GPT function calling tester.
        
        Args:
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.functions = DrivingBehaviorFunctions()
        
        # Define function schemas for GPT
        self.function_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_red_light_violation",
                    "description": "Analyze if a red light violation occurred based on vehicle behavior",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vehicle_position": {
                                "type": "string",
                                "description": "Position of vehicle relative to intersection",
                                "enum": ["approaching", "at_line", "entering", "past_intersection"]
                            },
                            "traffic_light_status": {
                                "type": "string", 
                                "description": "Current status of traffic light",
                                "enum": ["red", "yellow", "green"]
                            },
                            "intersection_crossed": {
                                "type": "boolean",
                                "description": "Whether vehicle crossed the intersection"
                            },
                            "timestamp": {
                                "type": "string",
                                "description": "When the event occurred"
                            }
                        },
                        "required": ["vehicle_position", "traffic_light_status", "intersection_crossed", "timestamp"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "analyze_stop_sign_violation",
                    "description": "Analyze if a stop sign violation occurred based on vehicle behavior",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vehicle_speed": {
                                "type": "number",
                                "description": "Speed of vehicle at stop sign in mph"
                            },
                            "stop_sign_present": {
                                "type": "boolean",
                                "description": "Whether a stop sign is present"
                            },
                            "complete_stop_made": {
                                "type": "boolean", 
                                "description": "Whether vehicle came to complete stop"
                            },
                            "location": {
                                "type": "string",
                                "description": "Location of the stop sign"
                            }
                        },
                        "required": ["vehicle_speed", "stop_sign_present", "complete_stop_made", "location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_traffic_regulation", 
                    "description": "Get traffic regulation information for a specific violation type",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "violation_type": {
                                "type": "string",
                                "description": "Type of violation",
                                "enum": ["red_light", "stop_sign", "speeding", "reckless_driving"]
                            },
                            "state": {
                                "type": "string",
                                "description": "State for which to get regulations",
                                "default": "Washington"
                            }
                        },
                        "required": ["violation_type"]
                    }
                }
            }
        ]
    
    def _call_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call the appropriate function based on function name and arguments.
        
        Args:
            function_name: Name of function to call
            arguments: Arguments to pass to function
            
        Returns:
            Function result
        """
        if function_name == "analyze_red_light_violation":
            return self.functions.analyze_red_light_violation(**arguments)
        elif function_name == "analyze_stop_sign_violation":
            return self.functions.analyze_stop_sign_violation(**arguments)
        elif function_name == "get_traffic_regulation":
            return self.functions.get_traffic_regulation(**arguments)
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    def test_driving_scenario(self, scenario_description: str) -> Dict[str, Any]:
        """
        Test GPT function calling with a driving scenario description.
        
        Args:
            scenario_description: Natural language description of driving scenario
            
        Returns:
            Dict containing test results and function call details
        """
        try:
            # Create chat completion with function calling
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 as GPT-5 doesn't exist yet
                messages=[
                    {
                        "role": "system",
                        "content": "You are a traffic violation analysis system. Analyze driving scenarios and use the provided functions to determine if violations occurred. Always call appropriate functions based on the scenario described."
                    },
                    {
                        "role": "user", 
                        "content": f"Analyze this driving scenario: {scenario_description}"
                    }
                ],
                tools=self.function_schemas,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            function_results = []
            
            # Process function calls if any
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Call the actual function
                    function_result = self._call_function(function_name, function_args)
                    
                    function_results.append({
                        "function_name": function_name,
                        "arguments": function_args,
                        "result": function_result
                    })
            
            return {
                "scenario": scenario_description,
                "gpt_response": message.content,
                "function_calls": function_results,
                "success": True
            }
            
        except Exception as e:
            return {
                "scenario": scenario_description,
                "error": str(e),
                "success": False
            }
    
    def run_test_suite(self) -> List[Dict[str, Any]]:
        """
        Run a comprehensive test suite of driving scenarios.
        
        Returns:
            List of test results
        """
        test_scenarios = [
            "A vehicle approaches an intersection with a red traffic light at 2:30 PM. The vehicle is at the stop line when the light turns red, but the driver continues through the intersection.",
            
            "A car is traveling at 8 mph as it approaches a stop sign at the corner of Main St and 1st Ave. The driver slows down but does not come to a complete stop before proceeding.",
            
            "At 10:15 AM, a vehicle completely stops at a red light and waits for it to turn green before proceeding through the intersection.",
            
            "A driver approaches a stop sign at a residential intersection, comes to a complete stop for 3 seconds, looks both ways, then proceeds safely.",
            
            "A vehicle enters an intersection when the traffic light is yellow and crosses completely before it turns red. The vehicle was already past the stop line when the light changed.",
        ]
        
        results = []
        for scenario in test_scenarios:
            print(f"\nTesting scenario: {scenario[:50]}...")
            result = self.test_driving_scenario(scenario)
            results.append(result)
            
            if result["success"]:
                print(f"✓ Success - {len(result.get('function_calls', []))} function calls made")
            else:
                print(f"✗ Failed - {result.get('error', 'Unknown error')}")
        
        return results

def main():
    """
    Main function to run GPT function calling tests.
    """
    print("🚗 GPT Function Calling Test for Driving Behavior Analysis")
    print("=" * 60)
    
    try:
        # Initialize tester
        tester = GPTFunctionCallingTester()
        print("✓ GPT Function Calling Tester initialized")
        
        # Run test suite
        print("\n📋 Running test suite...")
        results = tester.run_test_suite()
        
        # Print summary
        print("\n📊 Test Results Summary:")
        print("=" * 40)
        
        successful_tests = sum(1 for r in results if r["success"])
        total_function_calls = sum(len(r.get("function_calls", [])) for r in results if r["success"])
        
        print(f"Total tests: {len(results)}")
        print(f"Successful tests: {successful_tests}")
        print(f"Failed tests: {len(results) - successful_tests}")
        print(f"Total function calls made: {total_function_calls}")
        
        # Print detailed results
        print("\n📝 Detailed Results:")
        for i, result in enumerate(results, 1):
            print(f"\nTest {i}: {result['scenario'][:50]}...")
            if result["success"]:
                print(f"  Status: ✓ Success")
                if result.get("function_calls"):
                    for fc in result["function_calls"]:
                        violation_detected = fc["result"].get("violation_detected", False)
                        violation_type = fc["result"].get("violation_type", "unknown")
                        print(f"  Function: {fc['function_name']}")
                        print(f"  Violation: {'YES' if violation_detected else 'NO'} ({violation_type})")
            else:
                print(f"  Status: ✗ Failed - {result.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    main()