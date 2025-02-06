# MM-RLHF Eval Documentation

Welcome to the docs for `mmrlhf-eval`! Majority of this documentation is adapted from [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)

### **[2025/02/10]** 🔥

**This repository serves as the evaluation suite for the MM-RLHF(https://github.com/yfzhang114/MMPreference) project.** This project is built upon the lmms_eval framework. We have established a dedicated *"Hallucination and Safety Tasks"* category, incorporating three key benchmarks - *AMBER, MMHal-Bench, and ObjectHallusion.* **Additionally, we introduce our novel MM-RLHF-SafetyBench task, a comprehensive safety evaluation protocol specifically designed for MLLM.** Detailed specifications of other tasks are documented in [current_tasks](current_tasks.md).

- [AMBER](https://github.com/junyangwang0410/AMBER) (amber)
- [POPE](https://github.com/RUCAIBox/POPE) (pope)
- [MMHal_Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench) (mmhal_bench)
- [HallusionBench](https://github.com/tianyi-lab/HallusionBench) (hallusion_bench_image)
- [Object HalBench](https://github.com/LisaAnne/Hallucination) (objecthallusion)
  - object_hallucination
- [MM-RLHF-SafetyBench](lmms_eval/tasks/mm-rlhf-safetybench)
  - ```Adv_target``` (Adversarial Attack: The ratio of model descriptions containing adversarial image content.)
  - ```Adv_untarget``` (Adversarial Attack: The ratio of model descriptions containing original image content.)
  - ```Crossmodel_ASR``` (Cross-modal Jailbreak: Inducing multimodal models to deviate from their expected behavior and security constraints by combining text and image contexts related to the jailbreak task.)
  - ```Multimodel_ASR``` (Multimodal Jailbreak: Using a combination of various modalities (e.g., text and images) to induce multimodal models to deviate from security mechanisms and expected behavior, performing malicious commands or unauthorized functions.)
  - ```Typographic_ASR``` (Typographic Jailbreak: Converting malicious text instructions into images to perform jailbreak attacks, inducing multimodal models to deviate from their expected behavior and security constraints.)
  - ```Crossmodel_RtA``` (Cross-modal Jailbreak: Inducing multimodal models to deviate from their expected behavior and security constraints by combining text and image contexts for the jailbreak task.)
  - ```Multimodel_RtA``` (Multimodal Jailbreak: Using a combination of various modalities (e.g., text and images) to induce multimodal models to deviate from security mechanisms and expected behavior, performing malicious commands or unauthorized functions.)
  - ```Typographic_RtA``` (Typographic Jailbreak: Converting malicious text instructions into image format for jailbreak attacks, inducing multimodal models to deviate from their expected behavior and security constraints.)
  - ```Risk_identification``` (Identification Ability: The model's ability to correctly identify dangerous items and assess risk.)
  - ```Unsafes``` (Visual-Linguistic Safety: Ratio of model not rejecting output when the image contains harmful information.)
  - ```Safe_unsafes``` (Visual-Linguistic Safety: Ratio of model not rejecting output when the image is harmless but the instruction contains harmful content.)
  


## Table of Contents

* To learn about the command line flags, see the [commands](commands.md)
* To learn how to add a new moddel,  see the [Model Guide](model_guide.md).
* For a crash course on adding new tasks to the library, see our [Task Guide](task_guide.md).
* If you need to upload your datasets into correct HF format with viewer supported, please refer to [tools](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/pufanyi/hf_dataset_docs/tools)

