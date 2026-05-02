import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#1a1a2e')

points = [
    (58.1, 6.6, '#0 Base Model', '#6b7280', 150),
    (84.9, 5.7, '#1 Original SVEN', '#3b82f6', 120),
    (81.6, 5.7, '#2 No Contrastive', '#ef4444', 120),
    (86.0, 5.3, '#3 Single+UL', '#f59e0b', 120),
    (84.8, 6.2, '#4 Sec-Only', '#10b981', 200),
]

for x, y, label, color, size in points:
    ax.scatter(x, y, c=color, s=size, edgecolors='white', linewidths=0.8, zorder=5)

offsets = [(1.5, 0.15), (0, -0.25), (-5, 0.15), (0, -0.22), (-0.5, 0.18)]
for (x, y, label, color, _), (dx, dy) in zip(points, offsets):
    weight = 'bold' if '#4' in label else 'normal'
    star = ' ★' if '#4' in label else ''
    ax.annotate(f'{label}{star}\n({x}%, {y}%)', (x, y),
                xytext=(x+dx, y+dy), color=color, fontsize=9, fontweight=weight,
                ha='center', va='bottom')

ax.axhspan(5.9, 7.5, xmin=(83-50)/(92-50), xmax=1.0, alpha=0.08, color='#10b981')
ax.plot([83, 92, 92, 83, 83], [5.9, 5.9, 7.5, 7.5, 5.9], '--', color='#10b98140', linewidth=1)
ax.text(91.5, 7.1, 'Ideal Zone', color='#10b98188', fontsize=10, ha='right', va='top')

ax.annotate('', xy=(84.8, 6.2), xytext=(58.1, 6.6),
            arrowprops=dict(arrowstyle='->', color='#ffffff30', linestyle='dashed', lw=1))

ax.set_xlabel('Security Rate (%)', color='#aaa', fontsize=13, fontweight='bold')
ax.set_ylabel('pass@1 (%)', color='#aaa', fontsize=13, fontweight='bold')
ax.set_title('Security vs Utility Trade-off', color='#b794f4', fontsize=16, fontweight='bold', pad=15)
ax.set_xlim(50, 92)
ax.set_ylim(4.8, 7.2)
ax.tick_params(colors='#888')
ax.grid(True, alpha=0.1)
for spine in ax.spines.values():
    spine.set_color('#333')

plt.tight_layout()
plt.savefig('tradeoff_chart.png', dpi=150, facecolor=fig.get_facecolor())
print('Done! Saved to tradeoff_chart.png')
