#!/usr/bin/env python3
"""
Simple GPU monitoring script for basic status checks.
"""

import subprocess
import time
import argparse


def run_nvidia_smi():
    """Run nvidia-smi and return output."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error running nvidia-smi: {result.stderr}"
    except FileNotFoundError:
        return "nvidia-smi not found. Make sure NVIDIA drivers are installed."


def monitor_gpus(interval=2, duration=None):
    """Monitor GPUs with simple nvidia-smi output."""
    print("GPU Monitoring - Press Ctrl+C to stop")
    print("=" * 60)
    
    start_time = time.time()
    try:
        while True:
            # Clear screen (simple version)
            print("\033[2J\033[H")
            
            # Show timestamp
            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            # Show nvidia-smi output
            output = run_nvidia_smi()
            print(output)
            
            # Check duration
            if duration and (time.time() - start_time) >= duration:
                break
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def show_gpu_status():
    """Show current GPU status."""
    print("Current GPU Status:")
    print("=" * 40)
    output = run_nvidia_smi()
    print(output)


def main():
    parser = argparse.ArgumentParser(description="Simple GPU monitoring")
    parser.add_argument('--monitor', '-m', action='store_true', help='Start monitoring')
    parser.add_argument('--interval', '-i', type=int, default=2, help='Update interval (seconds)')
    parser.add_argument('--duration', '-d', type=int, help='Monitor duration (seconds)')
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_gpus(args.interval, args.duration)
    else:
        show_gpu_status()


if __name__ == "__main__":
    main()