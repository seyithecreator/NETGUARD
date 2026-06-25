"""Train both models and save to disk. Run once during build."""
from ml_engine import engine

print("Training models on NSL-KDD dataset…")
engine.train()
engine.save()
print("Done — model saved to models/trained.pkl")
