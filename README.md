# CSVtoBVH

FreeMoCap / MediaPipe のランドマークデータ（NumPy / CSV）を、BlenderやMixamo系リグで扱えるBVHへ変換するツールです。

主に `freemocap_to_bvh_plus.py` を使用します。`body_landmarks_to_bvh.py` はMediaPipe Body 33点だけを扱う簡易版です。

## 必要環境

- Python 3.9以降
- NumPy

```bash
python -m pip install numpy
```

## 入力

引数を省略した場合は、`input/` から次の名前を自動検出します。

| データ | 主なファイル名 |
| --- | --- |
| 身体 | `all_frame_name_xyz.npy`, `body_frame_name_xyz.npy` |
| 左手 | `left_hand_frame_name_xyz.npy` |
| 右手 | `right_hand_frame_name_xyz.npy` |
| 顔 | `face_frame_name_xyz.npy` |
| 重心 | `center_of_mass_frame_name_xyz.npy` |

NPYの基本形状は `frames x landmarks x 3` です。重心は `frames x 3` にも対応します。

CSVは以下の形式に対応します。

- `frame, name, x, y, z` のように1行が1ランドマーク
- `left_shoulder_x` のように列名へランドマーク名と軸を含むワイド形式

入力データや生成BVHはGitへ追加されない設定です。

## 基本的な使い方

既定名のファイルを `input/` に置いた場合:

```bash
python freemocap_to_bvh_plus.py
```

ファイルを明示する場合:

```bash
python freemocap_to_bvh_plus.py \
  --body input/all_frame_name_xyz.npy \
  --left-hand input/left_hand_frame_name_xyz.npy \
  --right-hand input/right_hand_frame_name_xyz.npy \
  --com input/center_of_mass_frame_name_xyz.npy \
  --output output/motion.bvh
```

Windows PowerShellでは、継続記号として `\` ではなくバッククォートを使うか、1行で実行してください。

## 主なオプション

| オプション | 説明 |
| --- | --- |
| `--rig-preset mixamo` | Mixamo風の階層・ボーン名で出力（既定） |
| `--rig-preset simple` | シンプルな階層で出力 |
| `--mixamo-prefix mixamorig:` | Mixamoボーン名の接頭辞 |
| `--rest-pose data` | 入力データの平均姿勢をレスト姿勢に使用（既定） |
| `--rest-pose tpose` | 計測した骨長からTポーズを生成 |
| `--rest-start N` | レスト姿勢の平均を開始する入力フレーム |
| `--rest-frames N` | レスト姿勢の平均に使うフレーム数 |
| `--start-frame N` | Nより前の入力フレームを読み飛ばす |
| `--fps 30` | BVHのフレームレート |
| `--scale 0.001` | 入力座標の倍率 |
| `--smooth N` | 身体ランドマークの平滑化窓。`1`で無効 |
| `--root-smooth N` | 腰・重心の平滑化窓 |
| `--hand-smooth N` | 手の平滑化窓 |
| `--face-smooth N` | 顔の平滑化窓 |
| `--use-hand-rotation` | 手ランドマークから手首方向を計算 |
| `--head-rotation-weight 0.0` | 顔から求める頭部回転の反映率 |
| `--joint-limit-mode soft` | 手首・足首の過回転を滑らかに制限 |
| `--debug-summary` | 読み込んだランドマークの概要を表示 |
| `--dump-joint-map PATH` | 関節対応表をJSON出力 |
| `--preview-first-frame PATH` | 最初の骨格をOBJ出力 |

すべての引数:

```bash
python freemocap_to_bvh_plus.py --help
```

## Blenderへの読み込み

現在の既定出力をBlenderへ読み込む際は、BVHインポート設定を以下にします。

- Forward: `-Z Forward`
- Up: `-Y Up`

この向きは、開発時に確認したBlender 4.5のリターゲット構成に合わせています。別のリグや座標系で向きが合わない場合は、次の変換オプションを使用できます。

```bash
python freemocap_to_bvh_plus.py --no-swap-yz
python freemocap_to_bvh_plus.py --flip-x
python freemocap_to_bvh_plus.py --flip-y
python freemocap_to_bvh_plus.py --flip-z
python freemocap_to_bvh_plus.py --rotate-x 90
python freemocap_to_bvh_plus.py --no-blender-retarget-compatible
```

## 簡易コンバーター

身体33点だけを変換する場合:

```bash
python body_landmarks_to_bvh.py input/body_frame_name_xyz.npy output/simple.bvh
```

## Blender補助スクリプト

`BlenderScript/` にはBlenderのScriptingワークスペースで実行する補助スクリプトがあります。

| ファイル | 用途 |
| --- | --- |
| `retarget_bvh_with_rest_correction.py` | レスト姿勢・ボーンロール差を補正してリターゲット |
| `retarget_legs_ik.py` | 脚をIKで補助リターゲット |
| `debug/inspect_active_rig.py` | 選択リグの構造をレポート |
| `debug/inspect_shoulder_retarget.py` | 腕・肩の回転問題を解析 |
| `legacy/retarget_left_hand_ik.py` | 旧・左腕IK検証用（通常は不使用） |

スクリプト冒頭のソース／ターゲット名を、Blenderシーン内のオブジェクト名に合わせてください。`debug/` は問題解析用ですが、将来のリグ差調査に使えるため残しています。

## ディレクトリ

```text
CSVtoBVH/
├── freemocap_to_bvh_plus.py
├── body_landmarks_to_bvh.py
├── BlenderScript/
│   ├── retarget_bvh_with_rest_correction.py
│   ├── retarget_legs_ik.py
│   ├── debug/
│   └── legacy/
├── input/
└── output/
```

`input/`、`output/`、Blenderの解析レポートには個人の計測データや生成物が入るため、`.gitignore` で除外しています。
