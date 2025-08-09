#!/usr/bin/env python3
"""
Multi-GPU test runner with performance monitoring and result collection.
"""

import sys
import argparse
import subprocess
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import yaml

import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiGPUTestRunner:
    """Test runner for multi-GPU training with performance monitoring."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.test_configs = {
            "single": project_root / "tests" / "debug_config.yaml",
            "2gpu": project_root / "tests" / "multigpu_2gpu_config.yaml", 
            "4gpu": project_root / "tests" / "multigpu_4gpu_config.yaml"
        }
        
        self.results = {}
        
    def get_gpu_info(self) -> Dict:
        """Get basic GPU information."""
        try:
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            gpu_info = []
            
            for i in range(gpu_count):
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i) if torch.cuda.is_available() else "Unknown"
                })
            
            return {
                "count": gpu_count,
                "gpus": gpu_info,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__
            }
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {"count": 0, "error": str(e)}
    
    def get_system_info(self) -> Dict:
        """Get basic system information."""
        return {
            "platform": sys.platform,
            "python_version": sys.version
        }
    
    def monitor_training(self, process: subprocess.Popen, test_name: str) -> Dict:
        """Simple training process monitoring."""
        start_time = time.time()
        
        logger.info(f"Monitoring {test_name} training...")
        
        # Just wait for process to complete
        process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "duration": duration,
            "return_code": process.returncode
        }
    
    def run_single_test(self, test_name: str, config_path: Path, timeout: int = 600) -> Dict:
        """Run a single training test."""
        logger.info(f"Running {test_name} test with config: {config_path}")
        
        if not config_path.exists():
            return {
                "success": False,
                "error": f"Config file not found: {config_path}",
                "duration": 0
            }
        
        # Create output directory for this test
        test_output_dir = self.results_dir / f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_output_dir.mkdir(exist_ok=True)
        
        # Update config to use test output directory
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['trainer_config']['output_dir'] = str(test_output_dir / "model_output")
        
        # Write modified config
        test_config_path = test_output_dir / "test_config.yaml"
        with open(test_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run training
        cmd = [
            sys.executable, "-m", "unsloth_multigpu.train",
            "--config", str(test_config_path)
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_root
            )
            
            # Monitor the process
            metrics = self.monitor_training(process, test_name)
            
            # Get output
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Save outputs
            with open(test_output_dir / "stdout.txt", 'w') as f:
                f.write(stdout)
            with open(test_output_dir / "stderr.txt", 'w') as f:
                f.write(stderr)
            
            success = process.returncode == 0
            if success:
                logger.info(f"✓ {test_name} completed successfully")
            else:
                logger.error(f"✗ {test_name} failed with return code {process.returncode}")
            
            return {
                "success": success,
                "duration": metrics["duration"],
                "return_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "output_dir": str(test_output_dir)
            }
            
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"✗ {test_name} timed out after {timeout} seconds")
            return {
                "success": False,
                "error": f"Timeout after {timeout} seconds",
                "duration": timeout
            }
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": 0
            }
    
    def calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate simple performance metrics from test results."""
        if not results.get("success"):
            return {}
        
        # Just return basic timing info
        return {
            "duration": results.get("duration", 0),
            "success": results.get("success", False)
        }
    
    def run_test_suite(self, test_types: List[str], timeout: int = 600) -> Dict:
        """Run a suite of tests."""
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"Available GPUs: {available_gpus}")
        
        if available_gpus == 0:
            logger.error("No CUDA GPUs available!")
            return {"error": "No CUDA GPUs available"}
        
        # Validate test types
        valid_tests = []
        for test_type in test_types:
            if test_type == "single":
                valid_tests.append(test_type)
            elif test_type == "2gpu" and available_gpus >= 2:
                valid_tests.append(test_type)
            elif test_type == "4gpu" and available_gpus >= 4:
                valid_tests.append(test_type)
            else:
                logger.warning(f"Skipping {test_type} test - insufficient GPUs")
        
        if not valid_tests:
            logger.error("No valid tests to run!")
            return {"error": "No valid tests to run"}
        
        # Record initial system state
        initial_state = {
            "gpu_info": self.get_gpu_info(),
            "system_info": self.get_system_info(),
            "timestamp": datetime.now().isoformat()
        }
        
        results = {"initial_state": initial_state, "tests": {}}
        
        # Run tests
        for test_type in valid_tests:
            config_path = self.test_configs[test_type]
            test_result = self.run_single_test(test_type, config_path, timeout)
            
            # Calculate performance metrics
            perf_metrics = self.calculate_performance_metrics(test_result)
            test_result["performance"] = perf_metrics
            
            results["tests"][test_type] = test_result
            
            # Brief pause between tests
            if test_type != valid_tests[-1]:
                logger.info("Pausing between tests...")
                time.sleep(10)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to: {results_file}")
        return results
    
    def print_summary(self, results: Dict):
        """Print test results summary."""
        print("\\n" + "="*60)
        print("MULTI-GPU TEST RESULTS SUMMARY")
        print("="*60)
        
        if "error" in results:
            print(f"ERROR: {results['error']}")
            return
        
        initial_state = results.get("initial_state", {})
        gpu_info = initial_state.get("gpu_info", {})
        
        print(f"System: {gpu_info.get('count', 0)} GPU(s)")
        if gpu_info.get("gpus"):
            for gpu in gpu_info["gpus"]:
                print(f"  GPU {gpu['id']}: {gpu['name']}")
        
        print("\\nTest Results:")
        print("-" * 40)
        
        tests = results.get("tests", {})
        for test_name, test_result in tests.items():
            status = "✓ PASS" if test_result.get("success") else "✗ FAIL"
            duration = test_result.get("duration", 0)
            
            print(f"{test_name:10} {status:8} {duration:6.1f}s")
            
            # Just show duration for successful tests
            if test_result.get("success"):
                print(f"           Duration: {duration:.1f}s")
        
        print("\\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Run multi-GPU training tests")
    parser.add_argument(
        "--tests", 
        nargs="+", 
        choices=["single", "2gpu", "4gpu", "all"],
        default=["all"],
        help="Test types to run"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per test in seconds (default: 600)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Expand "all" to all test types
    if "all" in args.tests:
        test_types = ["single", "2gpu", "4gpu"]
    else:
        test_types = args.tests
    
    runner = MultiGPUTestRunner(args.project_root)
    results = runner.run_test_suite(test_types, args.timeout)
    runner.print_summary(results)
    
    # Exit with error code if any tests failed
    if "error" in results:
        sys.exit(1)
    
    tests = results.get("tests", {})
    failed_tests = [name for name, result in tests.items() if not result.get("success")]
    
    if failed_tests:
        print(f"\\nFailed tests: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print("\\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()