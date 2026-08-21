# G1 29-DoF trajectory navigation

このタスクは既存の `velocity` タスクの歩容・正則化・sim-to-real設定を残し、
ランダムに生成したSE(2)軌道を追従するnavigationタスクです。

## 軌道生成

軌道は制御周期 `0.02 s` ごとに離散化し、4秒ごとに現在のロボット姿勢を
原点として生成します。

- 移動区間: 3秒
- 終端での停止保持: 1秒
- 曲率一定区間数: 1〜3
- 各区間の最短時間: 0.7秒
- 最小旋回半径: 0.15 m
- 直線区間の確率: 5%
- SE(2)速度: 0.20〜0.70
- 始動・停止ramp: 各0.8秒

各区間では曲率 `kappa` と進行距離 `s` から、数値的に安定なsinc形式で
局所姿勢を積分します。

```text
phi   = kappa * s
x     = s * sinc(phi / pi)
y     = 0.5 * s * phi * sinc(phi / (2 pi))^2
theta = phi
```

episodeの10%は停止軌道として生成し、静止動作も同時に学習します。

## 軌道観測

Actorには現在の推定自己位置から見た1秒後・2秒後の経路姿勢を追加します。
各姿勢は次の4要素で、合計8要素です。

```text
[x, y, cos(theta), sin(theta)] × 2
```

`x, y, theta`は現在のロボットbody座標系基準です。角度をcos/sinにすることで
`-pi/pi`境界の不連続を避けます。Criticには同じ形式の真値を渡します。

学習時のActor観測には、自己位置推定を再現する `0〜0.5 s` のepisode固定遅延を
適用します。平面位置には各制御stepでゼロ平均ガウスノイズを加え、通常99%は
標準偏差 `0.01 m`、残り1%は機器誤作動として標準偏差 `1.0 m` とします。
yawには位置ノイズを加えません。playでは遅延のみ無効化し、位置ノイズは維持します。

Actor観測は、従来のマーカー6要素を軌道8要素へ置き換えたため、腕目標14要素を
含めて120次元です。

## 指令と報酬

`twist`指令は最終ゴールへの比例制御ではなく、現在時刻の経路姿勢と接線速度から
生成します。接線速度へ現在の経路姿勢誤差を `tracking_gain=3.0` で補正し、body座標系へ
変換します。このため外乱から経路へ復帰できますが、最終点へ一直線に向かう指令は
生成されません。

navigation固有の `path_tracking` 報酬は、指定された次の2項をそのまま加算します。

```text
r_progress = s_closest(t) - s_closest(t - 1)

d_con = ||p - p_closest||^2
        + 2 r^2 (1 - cos(theta - theta_closest))
r_constellation = exp(-w_c * d_con)

r_path = r_progress + r_constellation
```

最近点は折れ線化した軌道の全区間へ射影して求めます。始点より後ろへずれた場合も
進行量を連続に評価できるよう、最初の曲率を使った後方延長を探索対象に含めます。
既定値は `r=1.0 m`, `w_c=0.2` です。負方向へ進むと `r_progress` も負になります。

velocityタスク由来の速度追従、姿勢、周期、足上げ、滑り、着地、関節平滑化の報酬は
引き続き有効です。腕姿勢はepisodeごとに上/下を同確率で選び、そのepisode中は固定します。

## 実行

```bash
python scripts/train.py Unitree-G1-Navigation-Flat \
  --env.scene.num-envs=4096
```

学習結果の確認:

```bash
python scripts/play.py Unitree-G1-Navigation-Flat \
  --checkpoint_file=logs/rsl_rl/g1_navigation/<run>/model_<iteration>.pt
```

入力が118次元から120次元へ変わったため、既存のnavigationチェックポイントは
読み込めません。新規にtrainしてください。

deploy用の修正済120次元設定は `navigation/v2_path` にあります。
修正後に新規trainしたONNXだけを使用してください。

```bash
mkdir -p deploy/robots/g1/config/policy/navigation/v2_path/exported
cp logs/rsl_rl/g1_navigation/<run>/policy.onnx deploy/robots/g1/config/policy/navigation/v2_path/exported/policy.onnx
```

Navigationへ入るたびにランダム経路を1本生成し、その終端をゴールとして保持します。
実軌跡は `navigation_pose.csv`、全目標経路は `navigation_target_path.csv` に出力され、
`scripts/plot_deploy_navigation.py`で重ねて確認できます。

主要な調整箇所は `TrajectoryCommandCfg` の次の値です。

- 参照時間: `reference_times`
- 軌道時間: `motion_duration`, `stop_hold_duration`
- 軌道形状: `num_segments_range`, `min_segment_duration`, `min_radius`,
  `straight_probability`, `curvature_exponent`
- 速度・追従補正: `se2_speed_range`, `characteristic_length`, `tracking_gain`
- 自己位置ノイズ: `localization_position_noise_std`,
  `localization_outlier_probability`, `localization_outlier_position_std`
- constellation: `path_tracking`報酬の`decay`, `constellation_radius`
