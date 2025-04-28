# Goal

We want to build a relatively small model, where it is realistic to train an "epoch" within 1 minute on modest hardware.

* Vocab Size: 4096
* Embedding Dim: 128
* Context Size: 64
* Hidden Layers: 6 (Let's assume hidden dimension is also 128 for simplicity, a common choice)
* Output Layer: Maps from the last hidden layer (128 dims) to Vocab Size (4096 dims)
* Data Size: 10 MB text


## Parameter Estimate

* Embedding Matrix: 4096 × 128 ≈ 0.52 Million
* Input Layer (Hidden Layer 1): (64 × 128) × 128 ≈ 1.05 Million
* Hidden-to-Hidden (5 layers): 5 × (128 × 128) ≈ 0.08 Million
* Output Layer: 128 × 4096 ≈ 0.52 Million

= 2.17 million parameters

## Memory Estimate

Assuming 32-bit floats (FP32), the model parameters require about 2.17 × 4 ≈ 8.7 MB of memory.

## Computation

Assuming, ~10MB intput text, ~5 bytes per token, roughly (10 × 1024 × 1024)/5 ≈ 2.1 million tokens

## Training Instances

There are approximately (InputTokens - ContextSize) training instances.

= (2.1M - 64) ≈ 2.1M training instances

## FLOPS per Instance

The bulk of computation is in matrix multiplications. 

A rough over estimate for a forward + backward pass is often around 2 × (Parameters × ContextSize)

= 2 × 2.17M × 64 ≈ 277 Million FLOPS

In practice, it should be closer to ~10 MFLOPS.

## FLOPS per Epoch

Total FLOPS is FlopsPerInstance × TrainingInstances

= 10M × 2.1M ≈ 21 TFLOPS

## FLOPS per Second

If the goal is to complete an Epoch in under 1 minute:

= 21 TFLOPS / 60 ≈ 350 MFLOPS/s

GPU stats are in FLOPS/s.

## Digial Ocean Options

Digital Ocean currently offers a NVIDIA L40S GPU for $1.57 / GPU / hour.

An NVIDIA L40S can sustain 91.6 TFLOPS

This is orders of magnitude more powerful than we need.

Theoretically, we should be able to train >1 Epoch per second, and very easily >1000 Epochs in an hour.

That is, the price to calculate an Epoch is well under 1/10th of 1 cent.
