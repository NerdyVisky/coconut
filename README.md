# COCONUT
## This codebase analyzes the pros and cons of latent reasoning using the recent work by Meta - [COCONUT](https://github.com/facebookresearch/coconut)

This is a fork from the official implementation of [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) for additional analysis

![coconut](assets/coconut.png)

### Note : To get started with using this repo checkout the [README](https://github.com/facebookresearch/coconut) in the offical implementation


## Additional Experiments

This section details additional experiments conducted to further analyze the Coconut model.

### Experiment 1: Performance on GSM8k-Hard and GSM8k-Platinum and 
**Aim:** To evaluate the performance of both CoT and COCONUT techniques on the more challenging GSM8k-Hard and GSM8k-Platinum datasets.

**Design:** Design : Extract raw outputs — CoT traces and final answers, for each of the settings and derive accuracy statistics. Run evaluations of CoT and COCONUT on the GSM8k-Hard and GSM8k-Platinum evaluation sets. The checkpoints used for both Stage 0 (only CoT training) and Stage 1 (COCONUT training) are utilized from this [HuggingFace artifact](https://huggingface.co/Esther22/coconut_Reproduction) contributed by a [Esther22](https://huggingface.co/Esther22)

**Finding 1:** CoT is marginally better on GSM8k-Hard and GSM8k-Platinum

![CoT vs Coconut Planning and Execution](/images/gsm8khard_platinum.png)

### Experiment 2: Which setup benefits more from the introduction of simple tool-use?
**Aim:** As described in the recent [To-CoT-or-not-to-CoT](https://arxiv.org/abs/2409.12183) work, they divide current reasoning problems into two sequential steps - planning and execution. Hence the aim of this experiment is to compare and contrast the predictions made using CoT and COCONUT techniques for the GSM8K dataset to identify potential weaknesses in Coconut's reasoning process. Thereby we aim to evaluate performance based purely on planning with execution outsourced. 

Additionally, to assess the impact of incorporating tool solvers on the performance of each method.

**Design:**   Develop a primitive rule-based arithmetic solver that corrects the execution step from the reasoning chain and derives accurate output. Measure the performance gains achieved by integrating tool solvers for each method.
![Tool Use Schematic](/images/tool_diagram.png)

**Finding 2:** Tool solvers improve the performance of CoT more significantly than they improve the performance of COCONUT on these harder datasets.

![Tool Use Comparison Table](/images/gsm_tool_effect.png)

The following confusion matrix better describes the improvement by tool use. It is evident how tool use improves CoT way more than COCONUT due to the unavailabilty of percievable reasoning traces in the latter. (Left is CoT, Right is COCONUT)

![CoT_ConfMatrix_GSM8kHard](/images/COT_matrix.png)
![COCONUT_ConfMatrix_GSM8kHard](/images/COCONUT_matrix.png)


## Future Work

*   Investigate training more advanced reasoning models, such as DeepSeek distilled versions, to think latently.
*   Explore deriving an optimal combination of latent thinking, explicit CoT, and tool use for efficient test-time scaling.

## Contribution
*   Training other open-sourced small reasoning models with COCONUT and comparing their performance as well as number of output tokens with regular CoT approach
*   To rewrite the ```run.py``` script for single A100 GPU. (The current ```run_1gpu.py``` was my initial attempt at this but it has some bugs in it)
*   Evaluating on more reasoning benchmarks and trying to devise an optimal set of tool-calling, latent-thinking, and explciit reasoning for efficient test-time scaling.

## Citation
If you use this code base in your research, please cite our paper with the following BibTex entry:




## License
This code is released under the MIT license (see [LICENSE](LICENSE)).
