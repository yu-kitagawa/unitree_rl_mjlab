# Arm-enabled Navigation policy

This directory contains the deploy parameters for the 118-dimensional
arm-conditioned Navigation actor. After training, copy its ONNX file to:

```text
v1_arm/exported/policy.onnx
```

The G1 controller selects the newest Navigation directory that has an
`exported` directory. Until that model is installed, the existing 104-input
`v0` policy remains active and Navigation continues to work without arm pose
switching.
