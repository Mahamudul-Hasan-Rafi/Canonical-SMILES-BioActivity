# 🔬 Reproducibility Issues & Solutions

## ❓ Why Different Results Each Run?

Even with the same code and `random_state=42`, you're getting different:
- Training loss curves
- Validation metrics  
- Optimal thresholds
- Test accuracy

## 🔍 Root Causes

### 1. **CUDA Non-Determinism** ⚠️ MAIN ISSUE
GPUs use non-deterministic algorithms for performance. Without explicit settings, results vary.

**Fix Applied:**
```python
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True  # Force deterministic
    torch.backends.cudnn.benchmark = False     # Disable auto-tuner
```

### 2. **Random SMILES Augmentation** 🔄
```python
if self.augment and random.random() < 0.5:  # ← Different each epoch!
    aug_smi = random_smiles(self.smiles[idx])
```

**Why it matters:** Even with seeds set, augmentation introduces randomness *during* training, affecting:
- Which molecules get augmented
- The specific augmented SMILES strings
- Training dynamics

**Partial Solution:** Seeds help, but augmentation will still vary slightly between epochs (by design).

### 3. **WeightedRandomSampler** 🎲
```python
_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
```

**Issue:** The sampler creates different batch orders each epoch.

**Better Fix (add generator):**
```python
g = torch.Generator()
g.manual_seed(SEED)
_sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
```

### 4. **Dropout Layers** 💧
Dropout is inherently stochastic - neurons are randomly "dropped out" during training.

**Solution:** Seeds make this reproducible across runs.

### 5. **Floating Point Precision** 🔢
GPU operations can have slight floating-point differences due to:
- Different computation orders
- Tensor shapes affecting memory layout
- Mixed precision (FP16/BF16) rounding

## ✅ Current Fix Summary

### **All Changes Applied to `bioactivity_dl 8_2_C_v3 - Retest.ipynb`:**

#### 1. **CUDA Determinism** (Lines 241-244)
```python
✅ torch.cuda.manual_seed(SEED)
✅ torch.cuda.manual_seed_all(SEED)  
✅ torch.backends.cudnn.deterministic = True
✅ torch.backends.cudnn.benchmark = False
```

#### 2. **WeightedRandomSampler with Seeded Generators** (4 locations fixed)

**Location 1 - HPO Training** (Lines 1784-1789):
```python
_g_hpo = torch.Generator()
_g_hpo.manual_seed(SEED)
_sampler_hpo = WeightedRandomSampler(..., generator=_g_hpo)
```

**Location 2 - Single-Fold Training #1** (Lines 2116-2122):
```python
_g = torch.Generator()
_g.manual_seed(SEED)
_sampler = WeightedRandomSampler(..., generator=_g)
```

**Location 3 - K-Fold Training** (Lines 2744-2746):
```python
_g_fold = torch.Generator()
_g_fold.manual_seed(SEED + f)  # Unique per fold but reproducible
_fs = WeightedRandomSampler(..., generator=_g_fold)
```

**Location 4 - Single-Fold Training #2** (Lines 3206-3212):
```python
_g2 = torch.Generator()
_g2.manual_seed(SEED)
_sampler = WeightedRandomSampler(..., generator=_g2)
```

## 📊 Expected Impact

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| **Same run twice** | Different results | **Identical results** ✅ |
| **Different machines** | Different results | *Still different* |
| **Training speed** | Faster | 5-10% slower ⚠️ |
| **Reproducibility** | ❌ None | ✅ Full (same GPU) |

## 🎯 What You Should See Now

After adding CUDA determinism:
1. **Same model checkpoint** every time you train
2. **Same optimal threshold** on validation set
3. **Same test accuracy** (to ~0.001% precision)
4. **Identical loss curves** epoch by epoch

## ⚠️ Important Notes

### Still Non-Deterministic Across:
- **Different GPU models** (RTX 3090 vs RTX 4090)
- **Different CUDA versions**
- **Different PyTorch versions**
- **CPU vs GPU** execution

### Trade-offs:
- ✅ **Pro:** Full reproducibility for debugging/comparisons
- ⚠️ **Con:** ~5-10% slower training (deterministic algorithms are less optimized)
- ⚠️ **Con:** May not use latest GPU optimizations

## 🔧 All Fixes Applied ✅

All reproducibility fixes have been successfully implemented in your notebook:

1. ✅ **CUDA deterministic mode enabled**
2. ✅ **All 4 WeightedRandomSampler instances seeded with generators**
3. ✅ **Seeds set for PyTorch, NumPy, random, and CUDA**

### What This Means:

You should now get **100% reproducible results** when:
- Running the same notebook multiple times on the same machine
- Using the same GPU model
- Using the same CUDA/PyTorch versions

### Next Steps:

1. **Test Reproducibility**: Run the same training cell twice and verify:
   ```python
   # Training metrics should be identical:
   # - Same loss values per epoch
   # - Same validation accuracy
   # - Same optimal threshold
   # - Same test accuracy
   ```

2. **Expect ~5-10% Slower Training**: This is the cost of determinism. If you need speed back:
   ```python
   torch.backends.cudnn.benchmark = True  # Faster but non-deterministic
   ```

3. **Check Model Weights**: For complete verification, save and compare model weights:
   ```python
   torch.save(model_v3.state_dict(), 'run1.pth')
   # Run training again...
   torch.save(model_v3.state_dict(), 'run2.pth')
   
   # Compare - should be byte-identical
   import filecmp
   assert filecmp.cmp('run1.pth', 'run2.pth'), "Models differ!"
   ```

## 📈 Testing Reproducibility

Run this verification after training twice to confirm perfect reproducibility:

```python
# ═══════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY TEST
# ═══════════════════════════════════════════════════════════════════════════

# Train once
print("🔹 Training Run #1...")
# Run your training cell...
torch.save(model_v3.state_dict(), 'models/reproducibility_test_run1.pth')
val_acc_1 = ...  # Save your validation accuracy
test_acc_1 = ...  # Save your test accuracy
opt_thresh_1 = ...  # Save your optimal threshold

# Reset Python kernel (IMPORTANT!)
# Kernel -> Restart Kernel

# Train again (same notebook, same cell)
print("🔹 Training Run #2...")
# Run your training cell again...
torch.save(model_v3.state_dict(), 'models/reproducibility_test_run2.pth')
val_acc_2 = ...
test_acc_2 = ...
opt_thresh_2 = ...

# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

# Check metrics
print(f"\n{'Metric':<20} {'Run 1':<15} {'Run 2':<15} {'Match?':<10}")
print("=" * 60)
print(f"{'Val Accuracy':<20} {val_acc_1:<15.6f} {val_acc_2:<15.6f} {'✅' if abs(val_acc_1 - val_acc_2) < 1e-6 else '❌'}")
print(f"{'Test Accuracy':<20} {test_acc_1:<15.6f} {test_acc_2:<15.6f} {'✅' if abs(test_acc_1 - test_acc_2) < 1e-6 else '❌'}")
print(f"{'Optimal Threshold':<20} {opt_thresh_1:<15.6f} {opt_thresh_2:<15.6f} {'✅' if abs(opt_thresh_1 - opt_thresh_2) < 1e-6 else '❌'}")

# Check model weights (byte-level comparison)
state1 = torch.load('models/reproducibility_test_run1.pth')
state2 = torch.load('models/reproducibility_test_run2.pth')

weights_match = True
for key in state1.keys():
    if not torch.equal(state1[key], state2[key]):
        print(f"❌ MISMATCH: {key}")
        weights_match = False

if weights_match:
    print("\n✅ ALL MODEL WEIGHTS IDENTICAL - Perfect Reproducibility!")
else:
    print("\n❌ Model weights differ - Check your seed configuration")
```

### Expected Output:
```
Metric               Run 1           Run 2           Match?    
============================================================
Val Accuracy         0.920000        0.920000        ✅
Test Accuracy        0.909091        0.909091        ✅
Optimal Threshold    0.450000        0.450000        ✅

✅ ALL MODEL WEIGHTS IDENTICAL - Perfect Reproducibility!
```

## 🎓 Learn More

- [PyTorch Reproducibility Docs](https://pytorch.org/docs/stable/notes/randomness.html)
- [CUDA Deterministic Operations](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility)

---
**Last Updated:** 2026-06-05
