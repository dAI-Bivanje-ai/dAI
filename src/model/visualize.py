import json
import matplotlib.pyplot as plt

with open("models/history.json") as f:
    h = json.load(f)


epochs = range(1, len(h["train_loss"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))


ax1.plot(epochs, h["train_loss"], label="Train loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoha")
ax1.legend()

ax2.plot(epochs, h["train_acc"], label="Train acc")
ax2.plot(epochs, h["val_acc"], label="Val acc")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoha")
ax2.set_ylim(0, 1)
ax2.legend()

plt.tight_layout()
plt.savefig("models/training_plot.png", dpi=150)
plt.show()
