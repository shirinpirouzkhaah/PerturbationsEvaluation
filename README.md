# PerturbationsEvaluation

This is a replication package of the work "Consistent or Sensitive? Automated Code Revision Tools Against Semantics-Preserving Perturbations"

This work evaluates the consistency of five automated code revision (ACR) tools—all based on transformer model architectures—when exposed to semantics-preserving perturbations in Java code. The repository provides (1) the perturbation framework, (2) inference and prompting scripts for all evaluated models, and (3) all data and code necessary to reproduce the experimental results reported in the paper.
Please download the datasets related to this work from https://doi.org/10.5281/zenodo.17454435

## Repository Structure

### 1. PerturbationsFramework

This directory contains the Java framework used to automatically apply nine semantics-preserving perturbations to Java methods.
The source code is located at Located at: PerturbationsFramework/src/main/java/, Built using JavaParser v3.23.1. The main entry point is the JavaCodeAnalyzer.java file, which sequentially applies all available perturbations. Each perturbation is implemented as an individual Java class in PerturbationsFramework/src/main/java/PerturbationClasses/. Each file corresponds to one of the nine perturbations described in the paper (e.g., IfElseSwap.java, LoopExchange.java, etc.).

#### Note: Each perturbation is only applicable to methods that contain the relevant structure (e.g., IfElseSwap requires an if block).


### 2. VisualPerturbs

This folder provides visual examples of each perturbation type.
Each subfolder (one per perturbation, using the same names as in the paper) includes:

OriginalCode.png

OriginalRevision.png

PerturbedCode.png

PerturbedRevision.png

ReviewComment.png

These visuals illustrate how each perturbation transforms the source and revised Java methods.


### 3. LoRaTunedLLaMa

Contains the code used to evaluate the LoRA-tuned LLaMA model. Scripts in this folder reproduce inference and evaluation for the LoRA-tuned model.
This version of LLaMA was fine-tuned using parameter-efficient LoRA weights from the replication package of the paper “LLaMA-Reviewer: Advancing Code Review Automation with Large Language Models through Parameter-Efficient Fine-Tuning.”



### 4. Prompting_chatgpt_LLaMa3.3_Deepseekv3 
Includes the code and prompts used to evaluate ChatGPT-3.5-Turbo, LLaMA 3.3-70B, and DeepSeek V3 models. 
Inference performed with temperature = 0.2. All prompt templates used in the experiments are provided in this folder.


### 5. T5 

Contains inference code for the fine-tuned T5 model.
The fine-tuned weights correspond to the replication package of the paper "Using Pre-Trained Models to Boost Code Review Automation".


### 6. Results

The Results folder provides all scripts and data used to reproduce the quantitative results reported in the paper.

RQ1_S_Cap_Results/: Code to compute RQ1 results for the intersect dataset across all models.

RQ1_S_Theta_Results/: Scripts for computing RQ1 results for each model’s theta dataset.

RQ3_Delta_Results/: Code to calculate RQ3 performance deltas (consistency and sensitivity metrics) across perturbations.

RQ2_LogitRegression.R: R script implementing the multilevel mixed-effects logistic regression used to answer RQ2.


### Dependencies

Java 17+

JavaParser v3.23.1

Python 3.10+ (for inference and results scripts)

R 4.3+ (for regression analysis)

Required Python libraries:
transformers, torch, pandas, tqdm, scipy, statsmodels


### Reproducibility Guide

1- Apply Perturbations: Run the main class in PerturbationsFramework/src/main/java/JavaCodeAnalyzer.java to generate perturbed versions of Java methods.

2- Run Inference: Use the corresponding model folders (LoRaTunedLLaMa, T5, Prompting_chatgpt_LLaMa3.3_Deepseekv3) to test models on original and perturbed data.

3- Compute Results: Use scripts in the Results/ directory to reproduce RQ1–RQ3 results.

6- Run Regression Analysis: Execute RQ2_LogitRegression.R in R to reproduce mixed-effects logistic regression models and statistical tables.