# MM-RLHF Eval Documentation

Welcome to the docs for `mmrlhf-eval`! Majority of this documentation is adapted from [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)

### **[2025/02/10]** 🔥

**This repository serves as the evaluation suite for the MM-RLHF(https://github.com/yfzhang114/MMPreference) project.** This project is built upon the lmms_eval framework. We have established a dedicated *"Hallucination and Safety Tasks"* category, incorporating three key benchmarks - *AMBER, MMHal-Bench, and ObjectHallusion.* **Additionally, we introduce our novel MM-RLHF-SafetyBench task, a comprehensive safety evaluation protocol specifically designed for MLLM.** Detailed specifications of the MM-RLHF-SafetyBench are documented in [current_tasks](current_tasks.md).

## Table of Contents

* To learn about the command line flags, see the [commands](commands.md)
* To learn how to add a new moddel,  see the [Model Guide](model_guide.md).
* For a crash course on adding new tasks to the library, see our [Task Guide](task_guide.md).
* If you need to upload your datasets into correct HF format with viewer supported, please refer to [tools](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/pufanyi/hf_dataset_docs/tools)

