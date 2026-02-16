import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Set dark mode style
plt.style.use('dark_background')

# Load data
with open('outputs\history.json', 'r') as f:
    history = json.load(f)

with open('outputs\config.json', 'r') as f:
    config = json.load(f)

epochs = np.array(history['epoch'])
train_loss = np.array(history['train_loss'])
test_loss = np.array(history['test_loss'])
train_acc = np.array(history['train_acc'])
test_acc = np.array(history['test_acc'])

# Find key milestones with more precision
train_converge_idx = np.where(train_acc >= 0.99)[0][0] if np.any(train_acc >= 0.99) else -1
train_converge_epoch = epochs[train_converge_idx] if train_converge_idx != -1 else None

# Find grokking point - when test acc starts rapid ascent
# Look for the point where test accuracy crosses 50%
grok_idx = np.where(test_acc >= 0.5)[0][0] if np.any(test_acc >= 0.5) else -1
grok_epoch = epochs[grok_idx] if grok_idx != -1 else None

# Find when test acc reaches 99%
test_converge_idx = np.where(test_acc >= 0.99)[0][0] if np.any(test_acc >= 0.99) else -1
test_converge_epoch = epochs[test_converge_idx] if test_converge_idx != -1 else None

# Calculate memorization plateau duration
if train_converge_epoch and grok_epoch:
    memorization_duration = grok_epoch - train_converge_epoch
else:
    memorization_duration = None

# Calculate grokking speed (50% to 99% test accuracy)
if grok_idx != -1 and test_converge_idx != -1:
    grokking_speed = test_converge_epoch - grok_epoch
else:
    grokking_speed = None

# Find the sharpest increase in test accuracy
test_acc_diff = np.diff(test_acc)
max_increase_idx = np.argmax(test_acc_diff)
max_increase_epoch = epochs[max_increase_idx]
max_increase_value = test_acc_diff[max_increase_idx] * 100  # Convert to percentage points

print("=" * 70)
print("GROKKING ANALYSIS - (a + b) mod 113")
print("=" * 70)
print(f"\nModel Configuration:")
print(f"  • Architecture: {config['n_layers']}-layer transformer")
print(f"  • d_model: {config['d_model']}, heads: {config['n_heads']}, MLP: {config['d_mlp']}")
print(f"  • Parameters: {config['n_params']:,}")
print(f"  • Training fraction: {config['frac_train']*100:.0f}%")
print(f"  • Learning rate: {config['lr']}, Weight decay: {config['wd']}")

print(f"\n{'Phase Transitions:':=^70}")
print(f"\n1. MEMORIZATION PHASE")
print(f"   Train accuracy → 99%:     Epoch {train_converge_epoch:>6}")
print(f"   Test accuracy at this point: {test_acc[train_converge_idx]*100:>5.1f}%")

print(f"\n2. PLATEAU PHASE (Memorization without Generalization)")
print(f"   Duration:                  {memorization_duration:>6} epochs")
print(f"   Train accuracy:            {train_acc[train_converge_idx:grok_idx].mean()*100:>5.1f}%")
print(f"   Test accuracy (stagnant):  {test_acc[train_converge_idx:grok_idx].mean()*100:>5.1f}%")

print(f"\n3. GROKKING PHASE (Sudden Generalization)")
print(f"   Grokking begins:           Epoch {grok_epoch:>6} (test acc → 50%)")
print(f"   Test accuracy → 99%:       Epoch {test_converge_epoch:>6}")
print(f"   Grokking duration:         {grokking_speed:>6} epochs")
print(f"   Maximum jump:              {max_increase_value:>5.1f}% at epoch {max_increase_epoch}")

print(f"\n4. FINAL GENERALIZATION")
print(f"   Final train accuracy:      {train_acc[-1]*100:>5.1f}%")
print(f"   Final test accuracy:       {test_acc[-1]*100:>5.1f}%")
print(f"   Final train loss:          {train_loss[-1]:>7.4f}")
print(f"   Final test loss:           {test_loss[-1]:>7.4f}")

# Calculate statistics for different phases
pre_grok_test = test_acc[train_converge_idx:grok_idx]
post_grok_test = test_acc[test_converge_idx:]

print(f"\n{'Statistical Summary:':=^70}")
print(f"\nPre-Grokking Test Accuracy (epochs {train_converge_epoch}-{grok_epoch}):")
print(f"   Mean:  {pre_grok_test.mean()*100:.2f}%")
print(f"   Std:   {pre_grok_test.std()*100:.2f}%")
print(f"   Range: {pre_grok_test.min()*100:.2f}% - {pre_grok_test.max()*100:.2f}%")

print(f"\nPost-Grokking Test Accuracy (epochs {test_converge_epoch}+):")
print(f"   Mean:  {post_grok_test.mean()*100:.2f}%")
print(f"   Std:   {post_grok_test.std()*100:.2f}%")
print(f"   Min:   {post_grok_test.min()*100:.2f}%")

print("\n" + "=" * 70)

# Dark mode color scheme
color_train = '#5DADE2'      # Bright blue
color_test = '#EC7063'       # Bright red
color_grok = '#58D68D'       # Bright green
color_plateau = '#AEB6BF'    # Light gray
bg_dark = '#1a1a1a'          # Very dark background
grid_color = '#404040'       # Dark gray for grid

# Create improved visualization
fig = plt.figure(figsize=(20, 10), facecolor=bg_dark)
gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)

# 1. Loss curves (log scale)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#0d0d0d')
ax1.semilogy(epochs, train_loss, color=color_train, linewidth=2.5, label='Train Loss', alpha=0.9)
ax1.semilogy(epochs, test_loss, color=color_test, linewidth=2.5, label='Test Loss', alpha=0.9)
ax1.axvline(train_converge_epoch, color=color_plateau, linestyle='--', linewidth=1.5, alpha=0.6, label='Train Converged')
ax1.axvline(grok_epoch, color=color_grok, linestyle='--', linewidth=1.5, alpha=0.8, label='Grokking Begins')
ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold', color='white')
ax1.set_ylabel('Cross-Entropy Loss (log scale)', fontsize=14, fontweight='bold', color='white')
ax1.set_title('Loss Curves — f(a,b) = (a + b) mod 113', fontsize=16, fontweight='bold', pad=15, color='white')
ax1.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, color=grid_color)
ax1.legend(loc='best', fontsize=11, framealpha=0.9, facecolor='#2a2a2a', edgecolor=grid_color)
ax1.set_xlim(0, epochs[-1])
ax1.tick_params(colors='white')
ax1.spines['bottom'].set_color(grid_color)
ax1.spines['top'].set_color(grid_color)
ax1.spines['left'].set_color(grid_color)
ax1.spines['right'].set_color(grid_color)

# 2. Accuracy curves with phase annotations
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#0d0d0d')
ax2.plot(epochs, train_acc * 100, color=color_train, linewidth=2.5, label='Train Accuracy', alpha=0.9)
ax2.plot(epochs, test_acc * 100, color=color_test, linewidth=2.5, label='Test Accuracy', alpha=0.9)

# Phase regions with darker, more visible colors
ax2.axvspan(0, train_converge_epoch, alpha=0.15, color='#3498db', label='Learning Phase')
ax2.axvspan(train_converge_epoch, grok_epoch, alpha=0.15, color='#e67e22', label='Memorization Plateau')
ax2.axvspan(grok_epoch, test_converge_epoch, alpha=0.15, color='#27ae60', label='Grokking Phase')
ax2.axvspan(test_converge_epoch, epochs[-1], alpha=0.15, color='#7f8c8d', label='Generalized')

# Key milestone markers
ax2.plot(train_converge_epoch, train_acc[train_converge_idx] * 100, 'o', 
         color=color_plateau, markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=5)
ax2.plot(grok_epoch, test_acc[grok_idx] * 100, 'o', 
         color=color_grok, markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=5)

# Annotations
ax2.annotate(f'Grok @ {grok_epoch}', 
            xy=(grok_epoch, test_acc[grok_idx] * 100),
            xytext=(grok_epoch + 300, 60),
            fontsize=12, fontweight='bold', color=color_grok,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d0d0d', alpha=0.9, edgecolor=color_grok, linewidth=2),
            arrowprops=dict(arrowstyle='->', color=color_grok, lw=2))

ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold', color='white')
ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='white')
ax2.set_title('Grokking — Delayed Generalization', fontsize=16, fontweight='bold', pad=15, color='white')
ax2.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, color=grid_color)
ax2.legend(loc='center left', fontsize=9, framealpha=0.9, ncol=2, facecolor='#2a2a2a', edgecolor=grid_color)
ax2.set_xlim(0, epochs[-1])
ax2.set_ylim(0, 105)
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color(grid_color)
ax2.spines['top'].set_color(grid_color)
ax2.spines['left'].set_color(grid_color)
ax2.spines['right'].set_color(grid_color)

# 3. Zoomed view of grokking transition
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#0d0d0d')
zoom_start = max(0, grok_idx - 20)
zoom_end = min(len(epochs), test_converge_idx + 20)
zoom_epochs = epochs[zoom_start:zoom_end]
zoom_test_acc = test_acc[zoom_start:zoom_end] * 100

ax3.plot(zoom_epochs, zoom_test_acc, color=color_test, linewidth=3, marker='o', 
         markersize=4, alpha=0.9, label='Test Accuracy')
ax3.axhline(50, color=color_grok, linestyle='--', linewidth=1.5, alpha=0.5, label='50% Threshold')
ax3.axhline(99, color='#52BE80', linestyle='--', linewidth=1.5, alpha=0.5, label='99% Threshold')
ax3.axvline(grok_epoch, color=color_grok, linestyle=':', linewidth=2, alpha=0.6)

# Highlight max increase point
ax3.plot(max_increase_epoch, test_acc[max_increase_idx] * 100, '*', 
         color='#F4D03F', markersize=20, markeredgecolor='white', markeredgewidth=1.5, zorder=5,
         label=f'Max Jump: +{max_increase_value:.1f}%')

ax3.set_xlabel('Epoch', fontsize=14, fontweight='bold', color='white')
ax3.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold', color='white')
ax3.set_title('Grokking Transition — Zoomed View', fontsize=16, fontweight='bold', pad=15, color='white')
ax3.grid(True, alpha=0.2, linestyle=':', linewidth=0.8, color=grid_color)
ax3.legend(loc='lower right', fontsize=11, framealpha=0.9, facecolor='#2a2a2a', edgecolor=grid_color)
ax3.fill_between(zoom_epochs, 0, zoom_test_acc, alpha=0.2, color=color_test)
ax3.tick_params(colors='white')
ax3.spines['bottom'].set_color(grid_color)
ax3.spines['top'].set_color(grid_color)
ax3.spines['left'].set_color(grid_color)
ax3.spines['right'].set_color(grid_color)

# 4. Test accuracy gradient (rate of change)
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#0d0d0d')
acc_gradient = np.gradient(test_acc * 100, epochs)
ax4.plot(epochs, acc_gradient, color='#BB8FCE', linewidth=2, alpha=0.8)
ax4.axhline(0, color='white', linestyle='-', linewidth=0.8, alpha=0.3)
ax4.axvline(grok_epoch, color=color_grok, linestyle='--', linewidth=2, alpha=0.6, label='Grokking Begins')
ax4.axvline(max_increase_epoch, color='#F4D03F', linestyle='--', linewidth=2, alpha=0.8, label='Peak Gradient')

# Fill positive gradient region
positive_gradient = np.where(acc_gradient > 0, acc_gradient, 0)
ax4.fill_between(epochs, 0, positive_gradient, alpha=0.3, color='#27ae60', label='Improving')

ax4.set_xlabel('Epoch', fontsize=14, fontweight='bold', color='white')
ax4.set_ylabel('Test Accuracy Gradient (% per epoch)', fontsize=14, fontweight='bold', color='white')
ax4.set_title('Rate of Generalization', fontsize=16, fontweight='bold', pad=15, color='white')
ax4.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, color=grid_color)
ax4.legend(loc='upper right', fontsize=11, framealpha=0.9, facecolor='#2a2a2a', edgecolor=grid_color)
ax4.set_xlim(0, epochs[-1])
ax4.tick_params(colors='white')
ax4.spines['bottom'].set_color(grid_color)
ax4.spines['top'].set_color(grid_color)
ax4.spines['left'].set_color(grid_color)
ax4.spines['right'].set_color(grid_color)

plt.savefig('outputs/improved_grokking_analysis_dark.png', dpi=300, bbox_inches='tight', facecolor=bg_dark)
print("\n✓ Saved: outputs/improved_grokking_analysis_dark.png")

print("\n" + "=" * 70)
print("Analysis complete! Generated improved visualization (dark mode).")
print("=" * 70)

