# Oral Cancer Classification: Training Results

## Model Comparison Summary

| Version | Configuration | Peak Cell AUC | Val Loss | Status |
|---------|---------------|---------------|----------|--------|
| **v1.0** | Partial weights, No normalization, No augmentation | 0.9505 | 13336.5 | Unstable |
| **v1.1** | Full weights, ImageNet Norm, Sync Augmentation | 0.9240 | Stable | **Robust** |

---

## Densenet121_v1.0 (Baseline)
- **Status:** Initial prototype.
- **Issues:** Only loaded first 3 blocks; missing normalization.
- **Peak Performance:** AUC 0.9505 at Epoch 10.
- **Observation:** High validation AUC was likely inflated by overfitting to unnormalized features, evidenced by the extreme validation loss.

---

## Densenet121_v1.1 (Optimized)
- **Configuration:** 
  - **Weights:** Correctly loaded all 4 Dense Blocks + Transitions from RadImageNet.
  - **Normalization:** Applied ImageNet mean/std.
  - **Augmentation:** Synchronized Horizontal/Vertical flips and 90-degree rotations.
  - **Checkpointing:** Best model saved based on Val AUC.
- **Peak Performance:** AUC 0.9240 at Epoch 6.
- **Execution Log (v1.1):**
```text
Epoch 1 Summary: Train Loss 0.5874, Val AUC 0.8084, Epoch Time: 333.30s
Epoch 2 Summary: Train Loss 0.3540, Val AUC 0.8671, Epoch Time: 361.38s
Epoch 3 Summary: Train Loss 0.3031, Val AUC 0.8616, Epoch Time: 397.21s
Epoch 4 Summary: Train Loss 0.2812, Val AUC 0.8875, Epoch Time: 420.32s
Epoch 5 Summary: Train Loss 0.2519, Val AUC 0.8957, Epoch Time: 407.91s
Epoch 6 Summary: Train Loss 0.2427, Val AUC 0.9240, Epoch Time: 354.59s (BEST)
Epoch 7 Summary: Train Loss 0.2196, Val AUC 0.9191, Epoch Time: 328.90s
...
```

## Observations & Next Steps
- **v1.1 Stability:** The training is now mathematically sound with a realistic validation loss.
- **Generalization:** The addition of data augmentation makes v1.1 more likely to perform well on the hidden test set than v1.0, despite the lower numerical AUC on the validation split.
- **Future tuning:** Implementing a Learning Rate Scheduler and increasing `SAMPLES_PER_EPOCH` could push v1.1 past the 0.95 mark.
