# Trajectory Navigation policy with corrected arm control

This configuration is for a Navigation policy retrained after the arm-up
tracking fix. Copy that 120-dimensional ONNX file to:

```text
v2_path/exported/policy.onnx
```

The wrist-pitch action scale in this directory includes the arm-up tracking
fix. Do not replace its ONNX file with a policy trained using the old scale.

The generated target path is written to `log/navigation_target_path.csv` and
the measured robot path to `log/navigation_pose.csv`.
