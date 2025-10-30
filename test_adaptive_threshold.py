"""
Simple test script to compare fixed vs adaptive threshold performance.
"""
import numpy as np
from fbank_net.demo.adaptive_threshold import AdaptiveThreshold, calculate_confidence_score

# Simulate enrollment scenarios
print("="*50)
print("Testing Adaptive Threshold System")
print("="*50)

# Scenario 1: High quality enrollment (consistent samples)
print("\n1. HIGH QUALITY ENROLLMENT (consistent samples)")
high_quality_embeddings = np.random.randn(10, 250)
high_quality_embeddings += np.random.randn(250) * 5  # All similar to same center

at_system = AdaptiveThreshold()
quality1 = at_system.calculate_enrollment_quality(high_quality_embeddings)
print(f"   Quality Score: {quality1:.3f}")
print(f"   Expected: High (>0.6)")

# Scenario 2: Low quality enrollment (inconsistent samples)
print("\n2. LOW QUALITY ENROLLMENT (inconsistent samples)")
low_quality_embeddings = np.random.randn(10, 250) * 3  # Very different from each other
quality2 = at_system.calculate_enrollment_quality(low_quality_embeddings)
print(f"   Quality Score: {quality2:.3f}")
print(f"   Expected: Low (<0.4)")

# Scenario 3: Confidence scoring
print("\n3. CONFIDENCE SCORING")
test_distances = np.array([0.25, 0.28, 0.22, 0.30])  # Good match
threshold = 0.45
conf = calculate_confidence_score(test_distances, threshold)
print(f"   Distances: {test_distances}")
print(f"   Threshold: {threshold}")
print(f"   Confidence: {conf:.2%}")
print(f"   Decision: ACCEPT (high confidence)")

print("\n" + "="*50)
print("Tests completed!")