"""Script to compare the original and reproduced Pareto fronts."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the images
original = mpimg.imread('/workspace/CrazyRL/results/v250529/original_pareto_front.png')
reproduced = mpimg.imread('/workspace/CrazyRL/results/v250529/pareto_front.png')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Display the original Pareto front
ax1.imshow(original)
ax1.set_title('Original Pareto Front', fontsize=16)
ax1.axis('off')

# Display the reproduced Pareto front
ax2.imshow(reproduced)
ax2.set_title('Reproduced Pareto Front', fontsize=16)
ax2.axis('off')

# Add a main title
plt.suptitle('Comparison of Pareto Fronts', fontsize=20)

# Adjust layout and save
plt.tight_layout()
plt.savefig('/workspace/CrazyRL/results/v250529/pareto_front_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison saved to /workspace/CrazyRL/results/v250529/pareto_front_comparison.png")