#!/usr/bin/env python3
"""
Docker Container Test Script for Fraud Detection API

This script tests all endpoints of the containerized fraud detection model
to ensure proper functionality and performance.

Author: ML Engineering Team
Date: November 2025
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
TIMEOUT = 30
RETRY_ATTEMPTS = 5

def wait_for_container(max_wait=60):
    """Wait for container to be ready."""
    print(f"⏳ Waiting for container to start (max {max_wait}s)...")

    for i in range(max_wait):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Container ready in {i+1} seconds")
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)

    print(f"❌ Container failed to start within {max_wait} seconds")
    return False

def test_health_endpoint():
    """Test the health check endpoint."""
    print("\n🏥 Testing /health endpoint...")

    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        latency = (time.time() - start_time) * 1000

        print(f"   Status Code: {response.status_code}")
        print(f"   Latency: {latency:.2f}ms")

        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

def test_model_info_endpoint():
    """Test the model info endpoint."""
    print("\n📊 Testing /model-info endpoint...")

    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/model-info", timeout=TIMEOUT)
        latency = (time.time() - start_time) * 1000

        print(f"   Status Code: {response.status_code}")
        print(f"   Latency: {latency:.2f}ms")

        if response.status_code == 200:
            data = response.json()
            model_info = data.get('model_info', {})
            print(f"   Algorithm: {model_info.get('algorithm')}")
            print(f"   PR-AUC: {model_info.get('performance_metrics', {}).get('pr_auc')}")
            print(f"   Features: {model_info.get('feature_count')}")
            return True
        else:
            print(f"   ❌ Model info failed: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Model info error: {e}")
        return False

def test_predict_endpoint():
    """Test the prediction endpoint with sample transaction."""
    print("\n🔮 Testing /predict endpoint...")

    # Sample transaction data
    sample_transaction = {
        "transaction": {
            "TransactionAmt": 150.00,
            "ProductCD": "W",
            "card1": 13553,
            "card2": 404.0,
            "card3": 150.0,
            "card4": "discover",
            "card5": 142.0,
            "card6": "credit",
            "addr1": 315.0,
            "addr2": 87.0,
            "dist1": 19.0,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "C1": 1.0,
            "C2": 1.0,
            "C4": 0.0,
            "C5": 0.0,
            "C6": 1.0,
            "C7": 0.0,
            "C8": 0.0,
            "C9": 1.0,
            "C10": 0.0,
            "C11": 1.0,
            "C12": 0.0,
            "C13": 1.0,
            "C14": 1.0,
            "D1": 14.0,
            "D2": 14.0,
            "D3": 13.0,
            "D4": 1.0,
            "D8": 1.0,
            "D10": 14.0,
            "D15": 13.0
        },
        "threshold": 0.5
    }

    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_transaction,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        latency = (time.time() - start_time) * 1000

        print(f"   Status Code: {response.status_code}")
        print(f"   Prediction Latency: {latency:.2f}ms")

        if response.status_code == 200:
            data = response.json()
            prediction = data.get('prediction', {})
            print(f"   Fraud Probability: {prediction.get('fraud_probability', 'N/A'):.4f}")
            print(f"   Is Fraud: {prediction.get('is_fraud', 'N/A')}")
            print(f"   Risk Level: {prediction.get('risk_level', 'N/A')}")
            print(f"   Model Version: {prediction.get('model_version', 'N/A')}")

            # Validate response structure
            required_fields = ['fraud_probability', 'is_fraud', 'risk_level']
            missing_fields = [field for field in required_fields if field not in prediction]

            if missing_fields:
                print(f"   ⚠️  Missing fields: {missing_fields}")
                return False

            return True
        else:
            print(f"   ❌ Prediction failed: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return False

def test_batch_predict_endpoint():
    """Test the batch prediction endpoint."""
    print("\n📦 Testing /predict/batch endpoint...")

    # Sample batch transactions
    batch_data = {
        "transactions": [
            {
                "TransactionAmt": 150.00,
                "ProductCD": "W",
                "card4": "visa",
                "card6": "credit"
            },
            {
                "TransactionAmt": 2500.00,
                "ProductCD": "C",
                "card4": "mastercard",
                "card6": "debit"
            }
        ],
        "threshold": 0.5
    }

    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        latency = (time.time() - start_time) * 1000

        print(f"   Status Code: {response.status_code}")
        print(f"   Batch Latency: {latency:.2f}ms")

        if response.status_code == 200:
            data = response.json()
            batch_prediction = data.get('batch_prediction', {})
            print(f"   Total Transactions: {batch_prediction.get('total_transactions', 'N/A')}")
            print(f"   Fraud Count: {batch_prediction.get('fraud_count', 'N/A')}")
            print(f"   Batch Size: {data.get('batch_size', 'N/A')}")
            return True
        else:
            print(f"   ❌ Batch prediction failed: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Batch prediction error: {e}")
        return False

def test_threshold_update():
    """Test threshold update endpoint."""
    print("\n🎯 Testing /threshold endpoint...")

    try:
        # Update threshold
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/threshold",
            json={"threshold": 0.3},
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        latency = (time.time() - start_time) * 1000

        print(f"   Status Code: {response.status_code}")
        print(f"   Update Latency: {latency:.2f}ms")

        if response.status_code == 200:
            data = response.json()
            print(f"   New Threshold: {data.get('new_threshold', 'N/A')}")
            return True
        else:
            print(f"   ❌ Threshold update failed: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Threshold update error: {e}")
        return False

def run_performance_test(num_requests=10):
    """Run performance test with multiple requests."""
    print(f"\n⚡ Running performance test ({num_requests} requests)...")

    sample_transaction = {
        "transaction": {
            "TransactionAmt": 150.00,
            "ProductCD": "W",
            "card4": "visa"
        }
    }

    latencies = []
    success_count = 0

    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json=sample_transaction,
                timeout=TIMEOUT
            )
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

            if response.status_code == 200:
                success_count += 1

        except Exception as e:
            print(f"   Request {i+1} failed: {e}")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        success_rate = (success_count / num_requests) * 100

        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Average Latency: {avg_latency:.2f}ms")
        print(f"   Min Latency: {min_latency:.2f}ms")
        print(f"   Max Latency: {max_latency:.2f}ms")

        return success_rate > 95  # 95% success rate threshold

    return False

def main():
    """Run all tests."""
    print("🧪 DOCKER CONTAINER API TESTING")
    print("=" * 50)
    print(f"Target URL: {BASE_URL}")
    print(f"Test Time: {datetime.now().isoformat()}")
    print()

    # Wait for container to be ready
    if not wait_for_container():
        print("❌ Container failed to start. Exiting...")
        sys.exit(1)

    # Run all tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Info", test_model_info_endpoint),
        ("Single Prediction", test_predict_endpoint),
        ("Batch Prediction", test_batch_predict_endpoint),
        ("Threshold Update", test_threshold_update),
        ("Performance Test", lambda: run_performance_test(5))
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "✅ PASS" if result else "❌ FAIL"))
        except Exception as e:
            results.append((test_name, f"❌ ERROR: {e}"))

    # Print summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)

    for test_name, result in results:
        print(f"{test_name:<20} {result}")

    # Overall result
    passed = sum(1 for _, result in results if "✅" in result)
    total = len(results)

    print()
    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
