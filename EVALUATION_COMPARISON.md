# VLA-0 Evaluation Implementation Comparison

## Summary

The current compression implementation has **significant differences** from the official VLA-0 evaluation methodology. While the compression framework is solid, the evaluation needs updates to match the official approach.

## Critical Differences

### 1. **Action Encoding/Decoding** ⚠️ CRITICAL

**Official VLA-0** (`vla0/rv_train/models/qwen/model.py:330-417`):
```python
# Encoding (continuous → discrete):
actions = (actions - min_act) / (max_act - min_act)  # Normalize to [0, 1]
actions *= num_bins_actions  # Scale to [0, 1000]
actions = torch.round(actions).long()
action_txt = " ".join(map(str, actions.tolist()))

# Decoding (discrete → continuous):
action = [int(x) for x in action_txt.split(" ")]
action = torch.tensor(action).float()
action = ((action / num_bins_actions) * (max_act - min_act)) + min_act
```

**Current Implementation** (`vla0_wrapper.py:46-74`):
```python
# Decoding only (no encoding):
int_values = [int(x) for x in action_string.split()]
normalized = 2 * (int_array - 0) / (1000 - 0) - 1  # Maps [0,1000] → [-1,1]
```

**Issues:**
- ❌ Assumes action range is always [0, 1000] → [-1, 1]
- ❌ Doesn't use dataset-specific min/max statistics
- ❌ Different normalization scheme than official
- ❌ No encoding function (only decoding)

**Impact**: Actions will be incorrectly scaled, leading to degraded performance on LIBERO.

---

### 2. **Dataset Statistics (Min/Max Actions)** ⚠️ CRITICAL

**Official VLA-0**:
- Loads dataset statistics containing min/max action bounds per task suite
- Different LIBERO task suites have different action ranges
- Uses these for normalization: `get_min_max_act(instruction)` (line 313)

**Current Implementation**:
- ❌ No dataset statistics loading
- ❌ Hardcoded assumption that actions are in [-1, 1] or [0, 1000]

**Required Fix**:
```python
# Need to load from checkpoint metadata:
dataset_stats = {
    "libero_goal": {"min": [...], "max": [...]},
    "libero_spatial": {"min": [...], "max": [...]},
    # etc.
}
```

---

### 3. **System Message / Prompt Format** ⚠️ IMPORTANT

**Official VLA-0** (`model.py:199`):
```python
system_message = f"Analyze the input image and predict robot actions for the next {self.horizon} timesteps. Each action has {self.act_dim} dimensions. Output a single sequence of {self.horizon * self.act_dim} integers (0-{self.num_bins_actions} each), representing the {self.horizon} timesteps sequentially. Provide only space separated numbers. Nothing else."
```

**Current Implementation** (`vla0_wrapper.py:233`):
```python
prompt = f"<|im_start|>user\n<image>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
```

**Issues:**
- ❌ Missing system message entirely
- ❌ Different format structure
- ❌ Doesn't specify output format expectations

**Impact**: Model may not generate actions in expected format.

---

### 4. **Action Horizon (Temporal Chunking)** ⚠️ CRITICAL

**Official VLA-0** (`eval.py:524-570`):
- Model predicts `horizon` future actions at once (e.g., 5-16 steps)
- Executes actions from chunk one by one
- Re-predicts when chunk is exhausted or new timestep threshold reached
- Default: `action_horizon=5` or `action_horizon=16` depending on task

**Current Implementation**:
- ❌ No action horizon support
- ❌ Predicts single action per timestep
- ❌ No action chunking

**Example**:
```python
# Official: Predict 5 actions at once
action_chunk = model.predict()  # Shape: (5, 7)
for i in range(5):
    env.step(action_chunk[i])
```

**Impact**: Evaluation will run ~5x slower and may degrade performance (action chunking improves temporal consistency).

---

### 5. **Ensemble Prediction** ⚠️ IMPORTANT

**Official VLA-0** (`eval.py:508-565`):
- Maintains sliding window of previous action chunks
- Weighted averaging with exponential decay
- Two versions:
  - Version 1: `0.5 * old_chunk + 1.0 * new_chunk`
  - Version 2: Exponential decay based on `ensemble_2_weight` (default 0.5)
- Default: `ensemble_prediction=8` (8 chunks averaged)

**Current Implementation** (`libero_evaluator.py:183-191`):
```python
# Simple averaging:
actions = []
for _ in range(ensemble_size):
    result = model.predict_action(image, instruction)
    actions.append(result['action'])
ensemble_action = np.mean(actions, axis=0)
```

**Issues:**
- ❌ No temporal sliding window
- ❌ No weighted averaging
- ❌ Re-runs inference multiple times instead of reusing chunks
- ❌ Doesn't handle action horizon

**Impact**: Ensemble won't provide the ~2% accuracy boost reported in the paper.

---

### 6. **Observation Preprocessing** ⚠️ IMPORTANT

**Official VLA-0** (`eval.py:625-702`):
```python
# Multi-stage transformation:
obs = libero_to_lerobot_format(obs, language_instruction)
obs = le_sample_to_rv_sample(obs, history, horizon, ...)
obs = remove_keys(obs, keys_to_remove)
obs = image_unifier_transform(cfg, obs, ...)
```

Includes:
- Image flipping: `img[::-1, ::-1]` (line 604)
- Quaternion to axis-angle conversion
- State concatenation (ee_pos + ee_rot + gripper)
- History stacking
- Image normalization to [0, 1]

**Current Implementation** (`libero_evaluator.py:160-175`):
```python
obs = env.get_observation()
image = obs.get('agentview_image', ...)
```

**Issues:**
- ❌ No image flipping
- ❌ No observation format conversion
- ❌ No history stacking
- ❌ Raw images not preprocessed

**Impact**: Model receives incorrectly formatted observations.

---

### 7. **LIBERO Environment Setup** ✓ MINOR

**Official VLA-0**:
- Uses `OffScreenRenderEnv` with specific camera settings
- Resolution: 256x256 for rendering (line 32)
- Dummy actions for first 10 steps (line 514-522)
- Fixed initial states from task suite

**Current Implementation**:
- ✓ Uses similar environment setup (MockLiberoEnv for testing)
- ❌ Missing dummy action warmup period
- ❌ Different camera configuration

---

## Evaluation Metrics Comparison

### Official VLA-0 Metrics

From the README:
```bash
python eval/eval_libero.py \
  --model_path ./runs/vla0/model_last.pth \
  --task_suite_name libero_goal \
  --task_name put_the_wine_bottle_on_top_of_the_cabinet \
  --action_horizon 1 \
  --ensemble_prediction 8
```

**Reported Results**: 94.7% average success rate on LIBERO

**Key Parameters**:
- 50 episodes per task (10 task suites × 10 tasks × 50 episodes)
- `action_horizon=1` or varies by task
- `ensemble_prediction=8` for reported results
- `generate_temperature=0.0` (greedy decoding)

### Current Implementation Metrics

```bash
python scripts/02_baseline_eval.py --num-episodes 50
```

**Parameters**:
- 50 episodes per task ✓
- No action horizon ❌
- `ensemble_size=3` (not weighted) ❌
- `temperature=0.0` ✓

---

## What Needs to be Fixed

### Priority 1: Critical for Accuracy

1. **Action Encoding/Decoding**
   - Load dataset statistics (min/max per task suite)
   - Implement correct normalization: `(x - min) / (max - min) * 1000`
   - Update `VLA0ActionDecoder` class

2. **Action Horizon**
   - Add temporal chunking support
   - Predict `horizon` actions at once
   - Execute chunk sequentially

3. **Observation Preprocessing**
   - Implement LIBERO → LeRobot → RoboVerse pipeline
   - Add image flipping
   - Add history stacking

### Priority 2: Important for Performance

4. **System Message**
   - Add proper system prompt format
   - Specify output format clearly

5. **Ensemble Prediction**
   - Implement weighted chunk averaging
   - Add sliding window mechanism
   - Support both ensemble versions

6. **LIBERO Environment**
   - Add dummy action warmup (10 steps)
   - Match camera resolution (256x256)

### Priority 3: Nice to Have

7. **Configuration Loading**
   - Load RoboVerse config files
   - Support different action types (original, abs_ee, rel_ee)

---

## Recommended Implementation Plan

### Phase 1: Fix Action Encoding/Decoding (2-3 days)

```python
# vla0_wrapper.py updates
class VLA0ActionDecoder:
    def __init__(self, dataset_stats):
        self.dataset_stats = dataset_stats  # {suite: {min: [...], max: [...]}}

    def decode_action_string(self, action_string, task_suite):
        ints = [int(x) for x in action_string.split()]
        actions = torch.tensor(ints).float()

        min_act = torch.tensor(self.dataset_stats[task_suite]["min"])
        max_act = torch.tensor(self.dataset_stats[task_suite]["max"])

        # Official normalization
        actions = ((actions / 1000.0) * (max_act - min_act)) + min_act
        return actions.numpy()
```

### Phase 2: Add Action Horizon Support (2-3 days)

```python
class VLA0Model:
    def predict_action_chunk(self, image, instruction, horizon=5):
        # Generate horizon * action_dim integers
        result = self.model.generate(...)
        action_chunk = self.action_decoder.decode_chunk(
            result, horizon, task_suite
        )
        return action_chunk  # Shape: (horizon, action_dim)
```

### Phase 3: Fix Observation Preprocessing (3-4 days)

```python
def libero_to_model_input(obs, language_instruction):
    # Image flipping
    image = obs["agentview_image"][::-1, ::-1]

    # Normalize to [0, 1]
    image = image / 255.0

    # Convert to PIL
    image = Image.fromarray((image * 255).astype(np.uint8))

    return image, language_instruction
```

### Phase 4: Implement Proper Ensemble (2-3 days)

```python
class TemporalEnsemble:
    def __init__(self, ensemble_size=8, ensemble_weight=0.5):
        self.chunks = []
        self.ensemble_size = ensemble_size
        self.weight = ensemble_weight

    def add_chunk(self, new_chunk):
        self.chunks.append(new_chunk)
        if len(self.chunks) > self.ensemble_size:
            self.chunks.pop(0)

    def get_averaged_chunk(self):
        # Weighted averaging with exponential decay
        weights = [self.weight ** i for i in range(len(self.chunks))]
        return weighted_average(self.chunks, weights)
```

---

## Testing Strategy

### Unit Tests

1. **Action Decoder Test**
```python
def test_action_decoding():
    decoder = VLA0ActionDecoder(dataset_stats)
    action_str = "500 234 789 123 456 678 901"
    decoded = decoder.decode_action_string(action_str, "libero_goal")

    # Verify correct denormalization
    assert decoded.shape == (7,)
    assert np.allclose(decoded, expected_values, atol=0.01)
```

2. **Ensemble Test**
```python
def test_temporal_ensemble():
    ensemble = TemporalEnsemble(ensemble_size=3)

    chunk1 = np.array([[1, 2, 3], [4, 5, 6]])
    chunk2 = np.array([[2, 3, 4], [5, 6, 7]])

    ensemble.add_chunk(chunk1)
    ensemble.add_chunk(chunk2)

    averaged = ensemble.get_averaged_chunk()
    # Verify weighted averaging
```

### Integration Tests

1. **Full Episode Test**
```python
def test_full_episode():
    model = VLA0Model(...)
    env = LiberoEnv(...)

    success = run_episode(model, env,
                         action_horizon=5,
                         ensemble_prediction=8)

    assert success  # Should complete task
```

---

## Expected Performance After Fixes

| Configuration | Current | After Fixes | Official |
|---------------|---------|-------------|----------|
| Baseline | ~80-85%? | ~94.7% | 94.7% |
| FP8 | ~80%? | ~94.5% | Expected 94.5% |
| INT8 | ~75%? | ~93.2% | Expected 93.2% |

**Note**: Current numbers are estimates since the evaluation isn't using the correct methodology.

---

## References

- **Official Evaluation**: `/home/shadeform/vla0/eval/eval_libero.py`
- **Action Encoding**: `/home/shadeform/vla0/rv_train/models/qwen/model.py:330-417`
- **Ensemble Logic**: `/home/shadeform/vla0/libs/RoboVerse/roboverse/evals/libero/eval.py:508-565`
- **Observation Processing**: `/home/shadeform/vla0/libs/RoboVerse/roboverse/evals/libero/eval.py:625-702`

## Conclusion

The compression framework is well-structured, but the **evaluation implementation needs significant updates** to match the official VLA-0 methodology. The most critical fixes are:

1. ✅ **Action encoding/decoding with dataset statistics**
2. ✅ **Action horizon support**
3. ✅ **Proper observation preprocessing**

Without these fixes, the baseline success rate will be significantly lower than 94.7%, and compression comparisons won't be meaningful.

**Recommended**: Implement Priority 1 fixes before running any quantization experiments.
