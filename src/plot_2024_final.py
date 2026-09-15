import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(12, 7))

balls_remaining = [30, 24, 23, 18, 14, 12, 6, 5, 1]
naive_prob = [89.40, 88.84, 86.02, 87.16, 56.38, 41.24, 16.04, 9.76, 0.07]
contextual_prob = [90.48, 91.95, 91.47, 86.36, 46.17, 48.14, 18.13, 3.27, 0.17]

ax.plot(balls_remaining, naive_prob, label="Leaky Industry Model (Scoreboard Only)", 
        color='#d32f2f', linestyle='--', marker='o', linewidth=2)
ax.plot(balls_remaining, contextual_prob, label="Rigorous Model (Micro-Matchup Context)", 
        color='#1976d2', linewidth=3, marker='s', markersize=8)

ax.set_xlim(31, 0) 
ax.set_ylim(0, 100)

ax.axhline(y=50, color='black', linestyle=':', alpha=0.6)
ax.axvline(x=14, color='grey', linestyle='-', alpha=0.3)

ax.annotate('The Blindspot:\nNaive model misses\nBumrah\'s lethality', 
            xy=(14, 56.38), xytext=(19, 65),
            arrowprops=dict(facecolor='#d32f2f', shrink=0.05, width=1.5, headwidth=8),
            fontsize=11, fontweight='bold', color='#d32f2f')

ax.annotate('The Alpha:\nContextual engine drops SA\nbelow 50% at Jansen Wicket', 
            xy=(14, 46.17), xytext=(12, 25),
            arrowprops=dict(facecolor='#1976d2', shrink=0.05, width=1.5, headwidth=8),
            fontsize=11, fontweight='bold', color='#1976d2')

ax.set_title("Data Leakage & Momentum Blindness in T20 Death Overs (2024 Final)", fontsize=15, fontweight='bold')
ax.set_xlabel("Balls Remaining in Chase", fontsize=12, fontweight='bold')
ax.set_ylabel("South Africa Win Probability (%)", fontsize=12, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)

plt.tight_layout()
plt.savefig('data/ssac_figure_1.png', dpi=300)
print("Saved publication graphic to data/ssac_figure_1.png")