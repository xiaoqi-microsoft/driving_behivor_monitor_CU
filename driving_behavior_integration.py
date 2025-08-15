"""
Example integration of GPT function calling with Azure Content Understanding
for driving behavior monitoring.

This example shows how to combine Azure Content Understanding video analysis
with GPT function calling for comprehensive driving violation detection.
"""

import json
from typing import Dict, Any, Optional
from test_gpt_function_calling import GPTFunctionCallingTester
from content_understanding_client import AzureContentUnderstandingClient

class DrivingBehaviorMonitor:
    """
    Integrated driving behavior monitoring system combining Azure Content Understanding
    with GPT function calling for comprehensive violation detection.
    """
    
    def __init__(
        self, 
        azure_endpoint: str,
        azure_api_version: str,
        openai_api_key: str,
        azure_token_provider: Optional[callable] = None,
        azure_subscription_key: Optional[str] = None
    ):
        """
        Initialize the driving behavior monitor.
        
        Args:
            azure_endpoint: Azure Content Understanding endpoint
            azure_api_version: Azure API version
            openai_api_key: OpenAI API key for GPT function calling
            azure_token_provider: Azure token provider (optional)
            azure_subscription_key: Azure subscription key (optional)
        """
        # Initialize Azure Content Understanding client
        self.azure_client = AzureContentUnderstandingClient(
            endpoint=azure_endpoint,
            api_version=azure_api_version,
            token_provider=azure_token_provider,
            subscription_key=azure_subscription_key
        )
        
        # Initialize GPT function calling tester
        self.gpt_tester = GPTFunctionCallingTester(api_key=openai_api_key)
    
    def analyze_driving_video(
        self, 
        video_file_path: str, 
        analyzer_id: str = "driving-behavior-analyzer"
    ) -> Dict[str, Any]:
        """
        Analyze driving video using Azure Content Understanding and GPT function calling.
        
        Args:
            video_file_path: Path to driving video file
            analyzer_id: Azure analyzer ID for video analysis
            
        Returns:
            Dict containing comprehensive analysis results
        """
        try:
            # Step 1: Analyze video with Azure Content Understanding
            print("🎥 Analyzing video with Azure Content Understanding...")
            analyze_response = self.azure_client.begin_analyze(analyzer_id, video_file_path)
            video_analysis = self.azure_client.poll_result(analyze_response)
            
            # Extract relevant content from video analysis
            video_content = self._extract_video_content(video_analysis)
            
            # Step 2: Use GPT function calling to analyze driving behavior
            print("🤖 Analyzing driving behavior with GPT function calling...")
            gpt_analysis = self.gpt_tester.test_driving_scenario(video_content)
            
            # Step 3: Combine results
            combined_results = {
                "video_analysis": video_analysis,
                "extracted_content": video_content,
                "gpt_analysis": gpt_analysis,
                "violations_detected": self._extract_violations(gpt_analysis),
                "recommendations": self._generate_recommendations(gpt_analysis)
            }
            
            return combined_results
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def analyze_traffic_regulations(
        self, 
        regulation_document_path: str,
        analyzer_id: str = "document-field-extraction"
    ) -> Dict[str, Any]:
        """
        Analyze traffic regulation documents with Azure Content Understanding.
        
        Args:
            regulation_document_path: Path to regulation document (PDF)
            analyzer_id: Azure analyzer ID for document analysis
            
        Returns:
            Dict containing extracted regulation information
        """
        try:
            print("📄 Analyzing traffic regulations document...")
            analyze_response = self.azure_client.begin_analyze(analyzer_id, regulation_document_path)
            regulation_analysis = self.azure_client.poll_result(analyze_response)
            
            # Extract red light and stop sign regulations
            extracted_regulations = self._extract_regulations(regulation_analysis)
            
            return {
                "document_analysis": regulation_analysis,
                "extracted_regulations": extracted_regulations,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def comprehensive_violation_check(
        self,
        video_file_path: str,
        regulation_document_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive violation check combining video analysis and regulation lookup.
        
        Args:
            video_file_path: Path to driving video
            regulation_document_path: Optional path to regulation document
            
        Returns:
            Dict containing comprehensive violation analysis
        """
        results = {
            "video_analysis": {},
            "regulation_analysis": {},
            "violation_summary": {},
            "final_recommendation": ""
        }
        
        # Analyze video
        video_results = self.analyze_driving_video(video_file_path)
        results["video_analysis"] = video_results
        
        # Analyze regulations if document provided
        if regulation_document_path:
            regulation_results = self.analyze_traffic_regulations(regulation_document_path)
            results["regulation_analysis"] = regulation_results
        
        # Generate comprehensive summary
        violations = []
        if video_results.get("violations_detected"):
            violations.extend(video_results["violations_detected"])
        
        results["violation_summary"] = {
            "total_violations": len(violations),
            "violations": violations,
            "severity_levels": [v.get("severity", "unknown") for v in violations]
        }
        
        # Generate final recommendation
        if violations:
            high_severity = any(v.get("severity") == "high" for v in violations)
            results["final_recommendation"] = (
                "IMMEDIATE ACTION REQUIRED" if high_severity 
                else "ISSUE WARNING OR CITATION"
            )
        else:
            results["final_recommendation"] = "NO VIOLATIONS DETECTED"
        
        return results
    
    def _extract_video_content(self, azure_analysis: Dict[str, Any]) -> str:
        """
        Extract relevant content from Azure video analysis for GPT processing.
        
        Args:
            azure_analysis: Raw Azure Content Understanding analysis
            
        Returns:
            String description of video content for GPT analysis
        """
        # This is a simplified extraction - in practice, you'd parse the Azure response
        # to extract specific information about traffic lights, stop signs, vehicle behavior, etc.
        
        if "result" in azure_analysis:
            # Extract summary or description from Azure results
            summary = azure_analysis["result"].get("summary", "")
            if summary:
                return summary
        
        # Fallback to mock content for demonstration
        return (
            "Video shows a vehicle approaching an intersection at 2:30 PM. "
            "The traffic light is red when the vehicle reaches the intersection. "
            "The vehicle continues through the intersection without stopping."
        )
    
    def _extract_regulations(self, azure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract regulation information from Azure document analysis.
        
        Args:
            azure_analysis: Raw Azure document analysis
            
        Returns:
            Dict containing extracted regulation information
        """
        # This would parse the Azure response to extract specific regulation text
        # For now, return mock regulations for demonstration
        return {
            "red_light_regulation": "RCW 46.61.055 - Vehicle operators must stop at red traffic signals",
            "stop_sign_regulation": "RCW 46.61.065 - Vehicle operators must come to complete stop at stop signs"
        }
    
    def _extract_violations(self, gpt_analysis: Dict[str, Any]) -> list:
        """
        Extract violations from GPT function calling results.
        
        Args:
            gpt_analysis: GPT analysis results
            
        Returns:
            List of detected violations
        """
        violations = []
        
        if gpt_analysis.get("function_calls"):
            for func_call in gpt_analysis["function_calls"]:
                result = func_call.get("result", {})
                if result.get("violation_detected"):
                    violations.append(result)
        
        return violations
    
    def _generate_recommendations(self, gpt_analysis: Dict[str, Any]) -> list:
        """
        Generate recommendations based on GPT analysis.
        
        Args:
            gpt_analysis: GPT analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if gpt_analysis.get("function_calls"):
            for func_call in gpt_analysis["function_calls"]:
                result = func_call.get("result", {})
                if result.get("recommendation") and result.get("recommendation") != "No action needed":
                    recommendations.append(result["recommendation"])
        
        return recommendations

def demo_integration():
    """
    Demonstration of the integrated driving behavior monitoring system.
    """
    print("🚗 Driving Behavior Monitoring System Demo")
    print("=" * 50)
    
    # Note: This is a demo - you would need actual Azure and OpenAI credentials
    print("Note: This demo shows the integration structure.")
    print("To run with real data, configure Azure and OpenAI credentials.")
    
    # Example usage (commented out as it requires real credentials):
    """
    monitor = DrivingBehaviorMonitor(
        azure_endpoint="your-azure-endpoint",
        azure_api_version="2024-12-01-preview", 
        openai_api_key="your-openai-key",
        azure_token_provider=your_token_provider
    )
    
    # Comprehensive analysis
    results = monitor.comprehensive_violation_check(
        video_file_path="path/to/driving_video.mp4",
        regulation_document_path="path/to/regulations.pdf"
    )
    
    print(f"Violations detected: {results['violation_summary']['total_violations']}")
    print(f"Recommendation: {results['final_recommendation']}")
    """
    
    print("\n✅ Integration demo complete. See GPT_FUNCTION_CALLING_TESTS.md for details.")

if __name__ == "__main__":
    demo_integration()