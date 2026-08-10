# G1 29-DoF object-front navigation

このタスクは既存の `velocity` タスクを低レベル歩行器として残し、頭部
カメラで観測したArUcoマーカーの前まで移動する目標条件付きタスクです。

## タスクの定義

- 物体はロボット前方の距離 `0.8–3.0 m`、方位 `±1.2 rad` の範囲で配置
  されます（学習初期はカリキュラムにより狭い範囲）。
- ArUco面はロボット側を向きます。
- 目標ロボット姿勢は物体から `0.7 m` 手前で、物体の方向を向く姿勢です。
- 目標は `4–8 s` ごとに更新されます。

頭部カメラ基準のActor観測 `marker_pose_camera` は次の6要素です。

```text
[x, y, z, distance, sin(relative_yaw), cos(relative_yaw)]
```

カメラ座標は `+x: 前、+y: 左、+z: 上` です。実機ではLiDAR自己位置推定と
ArUco姿勢推定から同じ量を生成してください。角度を `sin/cos` にしているのは
`-pi/pi` 境界で観測が不連続になるのを避けるためです。

学習時は自己位置推定・ArUco検出のレイテンシを再現するため、各envのreset時に
`0〜0.5 s` の観測遅延を独立にサンプルします。既定の制御周期は `0.02 s`
なので `0〜25 step` に相当し、そのenvのエピソード中は同じ遅延を保持します。
Actorの `marker_pose_camera` のみが遅延し、Criticには現在値を渡します。
play設定ではこの遅延を無効にしています。

自己位置推定の平面位置には、各env・各制御stepでゼロ平均ガウスノイズを
加えます。通常99%は標準偏差 `0.01 m`、機器誤作動を表す残り1%は標準偏差
`1.0 m` です。1%判定はx/y成分ごとではなく、一つの自己位置取得に対して
一度だけ行い、選ばれた標準偏差からx/yを独立にサンプルします。同じ誤差を
Actorの `marker_pose_camera` と目標速度指令へ共通適用し、距離成分はノイズ
適用後の位置から再計算します。Critic、報酬、評価指標は真値を使用します。
長さで指定されたノイズのためyawには適用しません。この自己位置ノイズは
学習・playの両方で有効です。制御周期50 Hzで取得ごとに1%を判定するため、
各envでは平均約2秒に1回の外れ値が発生します。

## 目標指令と報酬

目標誤差には比例制御を適用し、元のvelocityタスクと同じ
`(vx, vy, wz)` 指令へ変換します。このため、velocityの速度追従、姿勢、周期、
足上げ、滑り、着地、関節平滑化の各報酬はそのまま有効です。

論文の式(5)は `constellation_reward` として追加しています。半径 `r` の円形
constellationでは、

```text
d_con = ||p - p_goal||^2 + 2 r^2 (1 - cos(yaw - yaw_goal))
r_con = exp(-w_c d_con)
```

で、既定値は論文と同じ `r=1.0 m`, `w_c=0.2` です。位置と向きを別々に加算
するのではなく、一つの幾何誤差として同時に小さくします。

位置誤差 `0.08 m`、向き誤差 `0.20 rad` の両方を満たすと到達状態がラッチ
され、回復半径内では速度指令を完全にゼロへ固定します。位置到達後は
`goal_stillness` 報酬により、ベース並進・回転速度と全関節速度をゼロへ
近づけます。

球体の外では、比例制御で生成した並進速度に `0.35 m/s` の下限を設けて
いるため、目標直前でも減速し続けることはありません。実際に球体へ入った
stepで位置到達をラッチし、並進速度指令を直接ゼロへ切り替えます。到達後に
`0.10 m` より外へ流れた場合だけラッチを解除して再接近します。この
ヒステリシスにより、8 cm境界付近の微小振動では再発進しません。

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

主要な調整箇所は以下です。

- 停止距離: `GoalPoseCommandCfg.stand_off_distance`
- 目標範囲: `GoalPoseCommandCfg.Ranges.object_distance/object_bearing`
- 目標速度: `position_control_stiffness`, `heading_control_stiffness`,
  `min_approach_speed`, `position_recovery_tolerance`, `lin_vel_x`,
  `lin_vel_y`, `ang_vel_z`
- 自己位置ノイズ: `localization_noise_enabled`,
  `localization_position_noise_std`, `localization_outlier_probability`,
  `localization_outlier_position_std`
- constellation報酬: `decay`, `constellation_radius`, termの`weight`
