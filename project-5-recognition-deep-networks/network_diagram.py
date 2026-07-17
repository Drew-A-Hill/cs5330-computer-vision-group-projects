import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# Layer info: (label, shape_text, x_position, box_width, color)
layers = [
    ("Input", "1x28x28", 0.2, 1.2, "#4A90D9"),
    ("Conv1\n10 5x5 filters", "10x24x24", 1.8, 1.4, "#E07B39"),
    ("MaxPool1\n2x2 + ReLU", "10x12x12", 3.6, 1.4, "#5BAE5B"),
    ("Conv2\n20 5x5 filters", "20x8x8", 5.4, 1.4, "#E07B39"),
    ("Dropout\n50%", "20x8x8", 7.2, 1.2, "#C0C0C0"),
    ("MaxPool2\n2x2 + ReLU", "20x4x4", 8.8, 1.4, "#5BAE5B"),
    ("Flatten", "320", 10.6, 1.0, "#9B59B6"),
    ("Linear1\n+ ReLU", "50", 12.0, 1.2, "#D94A4A"),
    ("Linear2\n+ LogSoftmax", "10", 13.6, 1.4, "#D94A4A"),
    ("Output", "10 classes", 15.4, 0.0, "#4A90D9"),
]

box_height = 2.0
y_center = 3.0

for i, (label, shape, x, width, color) in enumerate(layers):
    if i < len(layers) - 1:
        rect = patches.FancyBboxPatch(
            (x, y_center - box_height / 2), width, box_height,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(x + width / 2, y_center + 0.15, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
        ax.text(x + width / 2, y_center - 0.55, shape, ha='center', va='center',
                fontsize=7, color='white', style='italic')
    else:
        ax.text(x, y_center, label + "\n" + shape, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)

    # Draw arrow to next layer
    if i < len(layers) - 1:
        next_x = layers[i + 1][2]
        arrow_start = x + width
        arrow_end = next_x
        if arrow_end > arrow_start:
            ax.annotate('', xy=(arrow_end, y_center), xytext=(arrow_start, y_center),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_title("CNN Architecture for MNIST Digit Recognition", fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("/Users/abhirambanda/cs5330-computer-vision-group-projects/project-5-recognition-deep-networks/network_diagram.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Diagram saved to network_diagram.png")
