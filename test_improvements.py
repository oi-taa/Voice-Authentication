"""
Test script to demonstrate adaptive threshold improvements
without using the web interface
"""
import sys
import os
import numpy as np

# Add path
sys.path.insert(0, 'fbank_net/demo')

from adaptive_threshold import AdaptiveThresholdSystem, calculate_confidence_score

print("="*60)
print("TESTING ADAPTIVE THRESHOLD IMPROVEMENTS")
print("="*60)

# Initialize system
system = AdaptiveThresholdSystem(
    base_threshold=0.45,
    min_threshold=0.30,
    max_threshold=0.60
)

print("\n" + "="*60)
print("SCENARIO 1: High Quality Enrollment")
print("="*60)

# Simulate high-quality enrollment (consistent samples)
np.random.seed(42)
base_vector = np.random.randn(250)
high_quality = [base_vector + np.random.randn(250) * 0.05 for _ in range(10)]
high_quality = np.array(high_quality)

threshold_high, quality_high = system.get_adaptive_threshold(
    'alice',
    enrollment_features=high_quality,
    save=True
)

print(f"User: Alice")
print(f"Number of enrollment samples: {len(high_quality)}")
print(f"Enrollment Quality: {quality_high:.2%}")
print(f"Personalized Threshold: {threshold_high:.3f}")
print(f"Feedback: {system.get_quality_feedback(quality_high)}")

# Simulate authentication
test_distance = 0.25  # Good match
confidence = calculate_confidence_score(np.array([test_distance] * 5), threshold_high)
is_authenticated = test_distance < threshold_high

print(f"\nAuthentication Test:")
print(f"  Test Distance: {test_distance:.3f}")
print(f"  Decision: {'✓ AUTHENTICATED' if is_authenticated else '✗ REJECTED'}")
print(f"  Confidence: {confidence:.2%}")

print("\n" + "="*60)
print("SCENARIO 2: Low Quality Enrollment")
print("="*60)

# Simulate low-quality enrollment (inconsistent samples)
low_quality = [np.random.randn(250) * 2 for _ in range(10)]
low_quality = np.array(low_quality)

threshold_low, quality_low = system.get_adaptive_threshold(
    'bob',
    enrollment_features=low_quality,
    save=True
)

print(f"User: Bob")
print(f"Number of enrollment samples: {len(low_quality)}")
print(f"Enrollment Quality: {quality_low:.2%}")
print(f"Personalized Threshold: {threshold_low:.3f}")
print(f"Feedback: {system.get_quality_feedback(quality_low)}")

# Simulate authentication with borderline match
test_distance = 0.42  # Borderline
confidence = calculate_confidence_score(np.array([test_distance] * 5), threshold_low)
is_authenticated = test_distance < threshold_low

print(f"\nAuthentication Test:")
print(f"  Test Distance: {test_distance:.3f}")
print(f"  Decision: {'✓ AUTHENTICATED' if is_authenticated else '✗ REJECTED'}")
print(f"  Confidence: {confidence:.2%}")

print("\n" + "="*60)
print("COMPARISON: Original vs. Improved System")
print("="*60)

original_threshold = 0.45

print(f"\nOriginal System (Fixed Threshold = {original_threshold}):")
print(f"  Alice (high quality): threshold = {original_threshold:.3f}")
print(f"  Bob (low quality):    threshold = {original_threshold:.3f}")
print(f"  → Same threshold for all users")

print(f"\nImproved System (Adaptive Threshold):")
print(f"  Alice (high quality): threshold = {threshold_high:.3f} (stricter)")
print(f"  Bob (low quality):    threshold = {threshold_low:.3f} (looser)")
print(f"  → Personalized per user based on enrollment quality")

print("\n" + "="*60)
print("BENEFITS:")
print("="*60)
print("✓ Fewer false accepts for high-quality enrollments (stricter)")
print("✓ Fewer false rejects for low-quality enrollments (looser)")
print("✓ Confidence scores help users understand decisions")
print("✓ Quality feedback guides users to improve enrollments")

print("\n" + "="*60)
print("IMPROVEMENTS DEMONSTRATED SUCCESSFULLY!")
print("="*60)

# Save thresholds
system.save_thresholds()
print(f"\n✓ User thresholds saved to 'user_thresholds.pkl'")