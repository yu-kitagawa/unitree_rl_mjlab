# G1 29-DoF trajectory navigation

このタスクは既存の `velocity` タスクの歩容・正則化・sim-to-real設定を残し、
ランダムに生成したSE(2)軌道を追従するnavigationタスクです。

## 軌道生成

軌道は制御周期 `0.02 s` ごとに離散化し、6秒ごとに現在のロボット姿勢を
原点として生成します。

- 移動区間: 3秒
- 終端での停止保持: 3秒
- 曲率一定区間数: 1〜3
- 各区間の最短時間: 0.7秒
- 最小旋回半径: 0.15 m
- 直線区間の確率: 5%
- SE(2)速度: 0.20〜0.70
- 後退軌道の確率: 25%（速度0.10〜0.35 m/s）
- 始動・停止ramp: 各0.8秒

各区間では曲率 `kappa` と進行距離 `s` から、数値的に安定なsinc形式で
局所姿勢を積分します。

```text
phi   = kappa * s
x     = s * sinc(phi / pi)
y     = 0.5 * s * phi * sinc(phi / (2 pi))^2
theta = phi
```

episodeの25%は停止軌道として生成し、静止動作も同時に学習します。
後退時も軌道上の移動距離は正の進捗として評価されるため、正しい後退を
`path_progress` が罰しません。

## 軌道観測

Actorには現在の推定自己位置から見た1秒後・2秒後の経路姿勢を追加します。
各姿勢は次の4要素で、合計8要素です。

```text
[x, y, cos(theta), sin(theta)] × 2
```

`x, y, theta`は現在のロボットbody座標系基準です。角度をcos/sinにすることで
`-pi/pi`境界の不連続を避けます。Criticには同じ形式の真値を渡します。
基準indexは経過時間ではなく、現在位置に最も近いサンプリング済み経路点から
求めます。1秒後・2秒後は、`reference_time / control_dt`から自動計算した
index offsetをその基準indexへ加えて取得します。終端を超える場合はゴールに固定します。

学習時のActor観測には、自己位置推定を再現する `0〜0.5 s` のepisode固定遅延を
適用します。平面位置には各制御stepでゼロ平均ガウスノイズを加え、通常99%は
標準偏差 `0.01 m`、残り1%は機器誤作動として標準偏差 `1.0 m` とします。
yawには位置ノイズを加えません。playでは遅延のみ無効化し、位置ノイズは維持します。

Actor観測は、経路姿勢8要素と腕目標14要素を含めて120次元です。

## 指令と報酬

`twist`指令は最終ゴールへの比例制御ではなく、現在位置に最も近い経路点の姿勢と
接線速度から生成します。接線速度へ最近経路点との姿勢誤差を `tracking_gain=3.0`
で補正し、body座標系へ
変換します。このため外乱から経路へ復帰できますが、最終点へ一直線に向かう指令は
生成されません。
始点のほぼゼロの速度で停止し続けるのを避けるため、始動rampの接線速度だけは
従来どおり経過時間で立ち上げ、終端の減速rampは最近点indexに対応する値を使います。

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
引き続き有効です。腕姿勢はepisodeごとに下げ姿勢と腕上げ姿勢群を50%ずつ
選び、腕上げ側では生成した各姿勢を一様にサンプリングします。目標はそのepisode中は
固定です。腕上げ時は左右の手首へ独立に0〜9.81 Nの下向き荷重を与えます。
停止中に傾きまたは角速度が大きくなった場合は、脚の姿勢固定報酬を弱め、
腕姿勢を保ったまま踏み直せるようにしています。

## 腕上げ姿勢ライブラリ

荷物を体の前で保持する想定の腕上げ姿勢を、次のコマンドで再生成できます。

```bash
python scripts/generate_g1_arm_carry_poses.py \
  --num-raised-poses 32 --seed 42
```

既定ではランダム姿勢の半数を左右ほぼ対称、半数を左右非対称として生成し、
次をすべて満たす候補だけを
`src/assets/motions/g1/arm_vel/carry_poses.npz`に保存します。

- 非対称姿勢は、右腕を鏡映して比較した左右関節角のRMS差が0.12 rad以上
- G1の各関節制限から既定で0.08 rad以上内側
- 左右の掌siteが体の前方で荷物を支えられる幅・高さ
- 腕下げ姿勢からの補間途中を含め、MuJoCo上で自己接触なし
- 他の採用姿勢との関節空間RMS距離が既定で0.10 rad以上

非対称姿勢の割合は`--asymmetric-pose-fraction`で変更できます。

姿勢一覧はGUIなしでも確認できます。

```bash
python scripts/view_g1_arm_pose_library.py --list
```

MuJoCo viewerでは全姿勢が2秒ごとに切り替わります。特定の姿勢を固定表示する場合は
`--pose-index`を指定します。

```bash
python scripts/view_g1_arm_pose_library.py
# seed 42の既定ライブラリでは、index 4が最初の非対称姿勢
python scripts/view_g1_arm_pose_library.py --pose-index 4
```

## 実行

```bash
python scripts/train.py Unitree-G1-Navigation-Flat \
  --env.scene.num-envs=4096
```

学習結果の確認:

```bash
python scripts/play.py Unitree-G1-Navigation-Flat \
  --checkpoint-file=logs/rsl_rl/g1_navigation/<run>/model_<iteration>.pt
```

deployは `navigation/v2_path` の120次元観測と対応するaction scaleを使用します。
入力次元だけで互換性を判断せず、最近点基準の経路観測・腕目標群・関節順序・
action scale/offsetが学習設定と一致するONNXとdeploy.yamlを一緒に使用してください。
チェックポイントを入れ替える場合:

```bash
mkdir -p deploy/robots/g1/config/policy/navigation/v2_path/exported
cp logs/rsl_rl/g1_navigation/<run>/policy.onnx deploy/robots/g1/config/policy/navigation/v2_path/exported/policy.onnx
```

学習時はランダム軌道を生成します。deploy時は外部から指定された相対ゴールに
向かう経路を生成し、その終端を保持します。起動は
[sim/real手順](../../../doc/sim_real_workflow.md)、指令は
[Navigationプロトコル](../../../doc/navigation_goals.md)を参照してください。
実軌跡は `navigation_pose.csv`、全目標経路は `navigation_target_path.csv` に出力され、
`scripts/plot_deploy_navigation.py`で重ねて確認できます。
同じスクリプトで腕14関節の目標角とencoder実測角を重ねたウィンドウも開き、
各関節の最新誤差、全腕のRMS誤差、最大絶対誤差を確認できます。

主要な調整箇所は `TrajectoryCommandCfg` の次の値です。

- 参照時間: `reference_times`
- 軌道時間: `motion_duration`, `stop_hold_duration`
- 軌道形状: `num_segments_range`, `min_segment_duration`, `min_radius`,
  `straight_probability`, `curvature_exponent`
- 速度・追従補正: `se2_speed_range`, `characteristic_length`, `tracking_gain`
- 後退: `reverse_probability`, `reverse_speed_range`
- 自己位置ノイズ: `localization_position_noise_std`,
  `localization_outlier_probability`, `localization_outlier_position_std`
- constellation: `path_tracking`報酬の`decay`, `constellation_radius`
