# dimm-analyzer

`dimm-analyzer` は DIMM (Differential Image Motion Monitor) 観測を解析するための
MVP コマンドラインツールです。SharpCap で記録した mono SER 動画から 2 つの DIMM
星像を検出し、それぞれに境界付き 2D Gaussian をフィットして、差分像運動、Fried
parameter `r0`、天文シーイングを出力します。

主な対応入力は SharpCap mono16 SER です。SER ヘッダーが単純な mono 形式であれば
mono8 も読み取ります。RGB、BGR、Bayer 形式の SER は MVP では明確なエラーとして
拒否します。

## インストール

作業ディレクトリをこのパッケージ直下にします。

```bash
cd dimm_program_ver2/dimm_analyzer
```

macOS では仮想環境を使う方法を推奨します。`dimm-analyze` コマンドも仮想環境の
`bin` に入り、PATH の問題を避けやすくなります。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

インストール確認:

```bash
python -m pip show dimm-analyzer
which dimm-analyze
dimm-analyze --help
```

`pip show` で `Name: dimm-analyzer`, `Version: 0.1.0` のように表示され、
`which dimm-analyze` が `.venv/bin/dimm-analyze` を指していれば正常です。

仮想環境を使わず macOS Command Line Tools 付属の `python3` に直接入れる場合:

```bash
python3 -m pip install -e ".[dev]"
```

CLT 付属の古い pip では、`pyproject.toml` 形式の editable install が失敗することがあります。
その場合は仮想環境を使うか、pip を更新してから再実行してください。

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install -e ".[dev]"
```

この方法で `which dimm-analyze` が見つからない場合、CLI script が user base の
`bin` に入っている可能性があります。次を一度実行してから確認してください。

```bash
export PATH="$(python3 -m site --user-base)/bin:$PATH"
which dimm-analyze
```

src layout のため、未インストール状態で `python3 -m dimm_analyzer.cli` を直接実行する
場合だけ `PYTHONPATH=src` が必要です。通常は editable install 後に
`dimm-analyze` を使ってください。

基本的な実行例:

パスを指定せずに起動すると、science SER、設定ファイル、出力ディレクトリを選択できます。
macOS では Finder 風の選択ダイアログを優先し、GUI が使えない場合は端末入力に
fallback します。

```bash
dimm-analyze
```

端末入力だけで選びたい場合:

```bash
dimm-analyze --picker-mode terminal
```

従来どおりパスを明示して、選択フローを出さずに実行することもできます。

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results \
  --zenith-deg 12.0
```

デフォルトでは `--output` は「親フォルダ」として扱われ、実際の解析結果は
SERファイル名 stem と同じサブフォルダに保存されます。上の例では出力先は
`./results/arcturus/` です。

従来どおり `--output` で指定した場所へ直接出したい場合だけ、
`--no-auto-output-name` を指定します。

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/arcturus_test \
  --zenith-deg 12.0 \
  --no-auto-output-name
```

既に同名の出力フォルダがある場合、上書き事故を防ぐため標準ではエラーになります。
既存フォルダへ書き込みたい場合は `--overwrite`、別名で衝突回避したい場合は
`--append-run-id` を使います。`--append-run-id` では `00_33_10_002/`,
`00_33_10_003/` のように連番が付きます。

```bash
dimm-analyze \
  --input ./data/00_33_10.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results \
  --zenith-deg 12.0 \
  --append-run-id
```

以降の例でも、`--no-auto-output-name` を付けない限り、`--output` は親フォルダです。

`--input` には SER ファイルだけでなく、SharpCap の capture folder も指定できます。
フォルダを指定した場合は `.ser` を検索します。標準ではサブフォルダも再帰的に検索し、
再帰検索を止めたい場合は `--no-recursive` を付けます。

```bash
dimm-analyze \
  --input ./data/2026-06-12_capture \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results \
  --zenith-deg 12.0
```

フォルダ内の SER が 1 個だけなら自動で採用します。複数ある場合の選び方は
`--ser-select` で指定します。

- `error`: 複数SERがあれば一覧を出して停止します。標準値です。
- `newest`: 更新時刻が最も新しい SER を解析します。
- `largest`: ファイルサイズが最も大きい SER を解析します。
- `all`: すべての SER を、結合せず個別に解析します。

```bash
dimm-analyze \
  --input ./data/2026-06-12_capture \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results \
  --zenith-deg 12.0 \
  --ser-select newest
```

`--ser-select all` の場合、各 SER は出力先の下に SER 名のサブフォルダを作って解析します。
この場合の `--output` は batch 全体の親フォルダです。

```text
results/batch_test/
  00_31_02/
  00_33_10/
  00_35_18/
  batch_summary.csv
  batch_manifest.json
```

SharpCap の設定メモや撮影条件ファイルらしい companion file は、各SER出力フォルダの
`input_metadata/` にコピーされます。対象は `.txt`, `.ini`, `.json`, `.yaml`, `.csv` と、
ファイル名に `Settings`, `Capture`, `Camera` を含むファイルです。SER本体はコピーしません。
`summary.json` には `input_path_original`, `input_path_resolved_ser`,
`input_was_directory`, `ser_selection_mode`, `companion_files_copied` を記録します。
さらに `input_ser_path`, `input_ser_filename`, `input_ser_stem`, `output_root`,
`output_dir`, `auto_output_name`, `input_manifest_path` も記録します。

ダーク補正ありの例:

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --dark ./data/dark.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/dark_runs \
  --zenith-deg 12.0
```

SER timestamp がない場合は、fallback 用の FPS を指定できます。

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/fps_runs \
  --zenith-deg 12.0 \
  --fps 120.0
```

解析する frame 範囲や表示内容は、次のオプションで調整できます。

- `--start-frame`: 解析開始 frame。0 始まりで、この frame を含みます。
- `--end-frame`: 解析終了位置。この frame は含みません。
- `--max-frames`: start/end で選択した範囲から解析する最大 frame 数。
- `--preview`: 選択範囲の先頭から最大 500 frame だけ解析します。
- `--verbose`: 完了時に `summary.json` の `warnings` を端末へ追加表示します。

診断用に orientation、ROI、large_jump rejection を変えて再解析できます。

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/orientation_runs \
  --zenith-deg 10.0 \
  --orientation-mode manual \
  --mask-angle-deg 90
```

ROI を標準の 25x25 で明示する例:

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/roi_runs \
  --zenith-deg 10.0 \
  --roi-size 25
```

ROI safety 診断の余裕幅を変える例:

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/roi_safety_runs \
  --zenith-deg 10.0 \
  --roi-size 25 \
  --roi-safety-margin 5
```

ROI safety 診断を出さない場合は `--disable-roi-safety-check` を使います。端に近い星像で
解析前にROIを自動縮小したい場合だけ、明示的に `--auto-shrink-roi-if-unsafe` を指定します。
この2つは同時には指定できません。

large_jump rejection を無効にする例:

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/quality_runs \
  --zenith-deg 10.0 \
  --disable-large-jump-rejection
```

飽和判定や dx/dy 外れ値判定を切り替える例:

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/quality_runs \
  --zenith-deg 10.0 \
  --saturation-level 65535 \
  --saturation-margin 100 \
  --max-relative-motion-deviation-px 10
```

比較のために一部の rejection を無効化することもできます。

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/rejection_runs \
  --zenith-deg 10.0 \
  --disable-saturation-rejection \
  --disable-relative-motion-rejection
```

orientation auto/manual、ROI 21/25/31、large_jump on/off を一括比較する場合:

```bash
dimm-analyze \
  --input ./data/arcturus.ser \
  --config ./configs/vmc260l_imx432.yaml \
  --output ./results/comparison_runs \
  --zenith-deg 10.0 \
  --comparison-suite
```

比較結果は、解決後のSER出力フォルダ内の `comparison_summary.csv` にまとまります。
上の例では `./results/comparison_runs/arcturus/comparison_summary.csv` です。

このプログラムは固定 FPS を仮定しません。時刻情報の優先順位は次のとおりです。

1. SER timestamp
2. CLI の `--fps`
3. YAML config の `timing.fps`
4. frame index のみ。物理的な時間軸がないため警告します。

## 設定

VMC260L/IMX432 向けのデフォルト設定は以下にあります。

```bash
configs/vmc260l_imx432.yaml
```

有効ピクセルスケールは、`camera.pixel_scale_arcsec_per_px` が指定されていれば
その値をそのまま使います。`null` の場合は次の式で計算します。

```text
base_pixel_scale_arcsec_per_px * binning
```

デフォルト設定では binning 1 で `0.643 arcsec/px`、binning 2 で
`1.286 arcsec/px` です。binning 2 以上も許容しますが、重心精度が落ちる可能性が
あるため CLI で警告します。

`zenith_deg` は config または CLI `--zenith-deg` で必ず明示してください。このツールは
天頂角 0 度を暗黙には仮定しません。

標準の解析 ROI は `25x25 px`、fallback ROI は `31x31 px` です。今回のように
FWHM が `4〜5 px` 程度で安定している場合、25x25 でも 2D Gaussian fit に十分な範囲を
確保できます。星像が撮影ROIの端から十分離れている場合は `--roi-size 31` も使えます。
`roi_out_of_bounds` が多い場合は `--roi-size 25` や `--roi-size 21` を試してください。
ただし ROI を小さくしすぎると、背景推定や fit 安定性が落ちる可能性があります。
ROI size は中心ピクセルを対称に扱うため、必ず奇数にしてください。

ROI safety check は、各星像中心から画像端までの最小距離を調べます。外れ値1点だけで
ROI推奨値が壊れないよう、absolute minimum と robust statistics を分けて記録します。

```text
half_roi_px = (roi_size_px - 1) / 2
required_margin_px = half_roi_px + safety_margin_px
```

`summary.json` には absolute minimum として `edge_margin_all_min_px`,
`edge_margin_min_frame_index`, `edge_margin_min_spot_id` を残します。ただし
`recommended_max_roi_size_px` はこの最小値ではなく、reliable population の p05 を標準で
使います。p01/p05/median は `edge_margin_all_*` と `edge_margin_reliable_*` の両方を
出力します。reliable population は、2つの星像が検出され、両方のfitが成功し、
`spot_tracking_failed`, `bad_relative_motion`, `nan_result` ではない frame です。

推奨ROIは次の値を比較できます。

- `recommended_max_roi_size_px_from_min`: absolute minimum ベース。診断用。
- `recommended_max_roi_size_px_from_p01`: reliable p01 ベース。
- `recommended_max_roi_size_px_from_p05`: reliable p05 ベース。
- `recommended_max_roi_size_px`: 通常は p05 ベースの推奨値。

判定は robust 統計と低margin割合を使います。`edge_margin_reliable_p05_px` が
`required_margin_px` 以上で、`roi_out_of_bounds_fraction` と低margin割合が十分小さければ
`safe` です。p05 が `required_margin_px` 未満だが `half_roi_px` 以上、または少数だけ
余裕不足なら `warning` です。p05 が `half_roi_px` 未満、多くの reliable frame が端に近い、
または `roi_out_of_bounds_fraction` が大きい場合は `unsafe` です。absolute minimum が
`half_roi_px` を下回るだけでは即 `unsafe` にせず、`edge_margin_absolute_outlier_detected`
を `true` にして robust 判定に従います。

`roi_safety_report.csv` には frame/spot ごとの `left/right/top/bottom_margin_px` と
`min_margin_px` に加え、`included_in_reliable_population`, `below_half_roi`,
`below_required_margin`, `suspected_outlier` が出ます。`roi_safety_timeseries.png` では
spot1/spot2 の `min_margin_px`、`half_roi_px`、`required_margin_px`、reliable p05、
below-half点、suspected outlier点、p05ベースの推奨ROIを確認できます。

`summary.json` の `recommended_max_roi_size_px` は、現在の星像位置と安全余裕から見た
推奨最大ROIです。`auto_shrink_if_unsafe` はデフォルトOFFです。再現性を優先する通常解析では、
まず警告と診断プロットを確認し、必要に応じて `--roi-size 21` / `25` / `31` を比較してください。
`auto_shrink_if_unsafe: true` の場合も、absolute minimum ではなく p05ベースの推奨ROIを使います。
撮影時ROIが狭すぎる場合の根本対策は、撮影ROIを広げるか、2つの星像をより中央に寄せることです。

ダーク補正は任意です。有効にした場合、ダーク SER から median dark frame を作成し、
science frame を float 化したあとに差し引きます。差し引き後の負値はクリップせず、
診断に使えるよう float のまま保持します。

## Orientation

`manual` は `orientation.mask_angle_deg` を使います。角度は画像 +x 軸から物理的な
開口ベースライン方向までを度で指定します。

`auto_pair` は検出された 2 星像の平均ペア角を使います。ただし星像ペア方向が物理的な
開口ベースライン方向と一致するとは限らないため、あくまで暫定推定です。

`auto_consistency` は 0〜180 度を探索し、longitudinal/transverse の block seeing が
最も整合する角度を選びます。信頼度が低い場合は `summary.json` と警告に記録します。
valid block が 3 未満の場合は `orientation_reliable=false` になり、seeing は参考値扱い
にしてください。

## 品質診断

`frame_fit_success_rate` は、全 frame のうち両方の星像 fit と品質判定を通過した割合です。
0.75 未満では選別バイアスの可能性があるため、`result_reliability` は caution/bad に
寄りやすくなります。

mono16 SER の飽和レベルは通常 `65535` です。`peak_raw_max` として ROI 内の最大値は
従来どおり記録しますが、saturated 判定には ROI 全体の最大値だけは使いません。
Gaussian fit 中心、または fit 前の検出中心から半径 `2.0 px` 以内の core で、
`saturation_threshold` 以上のピクセル数を数えます。

```text
saturation_threshold = saturation_level - saturation_margin
saturated if saturated_core_pixel_count >= 1
```

ROI 内に飽和ピクセルがあっても中心近傍でなければ、星像本体の飽和ではなく
`hot_pixel_or_roi_outlier` として除外します。`per_frame_fits.csv` には
`peak_raw_max1/2`, `peak_core_max1/2`, `saturated_core_pixel_count1/2`,
`saturated_roi_pixel_count1/2` を出力します。`summary.json` には
`saturated_core_frame_count`, `saturated_core_frame_fraction`,
`saturated_roi_pixel_outlier_count`, `peak_raw_max1/2`, `peak_core_max1/2` を記録します。

撮影時の peak ADU は mono16 ではおおよそ `20000〜45000` 程度を推奨します。
`60000` 以上は飽和や Gaussian fit 破綻の危険が高く、`peak_timeseries.png` と
`peak_histogram.png` で確認してください。saturated が多い場合はゲインまたは露光時間を
下げて再撮影するのが第一候補です。

`dx_dy_timeseries.png` で dx/dy が急に符号反転したり、大きく飛ぶ場合は、大気揺らぎでは
なく spot1/spot2 の入れ替わり、誤検出、飽和、ROI外れ、異常fitの可能性があります。
このプログラムは前回有効 frame からの spot tracking と、dx/dy の running median からの
ずれを使って `spot_tracking_failed` または `bad_relative_motion` として除外します。

FWHM が通常値から急落または急増する場合は `fwhm_outlier` として除外できます。
`fwhm_timeseries.png` では `bad_fwhm` / `fwhm_outlier` の点を別色で表示します。

`summary.json` の `quality_flags` と `result_reliability` を必ず確認してください。

- `good`: valid block、fit成功率、飽和率、orientation が十分良好。
- `caution`: 一部の条件が弱く、診断プロットで確認が必要。
- `bad`: valid block が少ない、fit成功率が低い、飽和が多い、外れ値が多いなど。

`result_reliability` が `bad` の場合、seeing 値は暫定値として扱ってください。

## 出力

付属設定例では `output.save_csv`, `output.save_json`, `output.save_plots` がすべて
`true` のため、出力ディレクトリには以下が作成されます。独自設定でいずれかを
`false` にした場合は、対応する CSV、`summary.json`、PNG は作成されません。
`input_manifest.json` と batch 用の台帳はこれらの設定に依存しません。

- `per_frame_fits.csv`
- `block_results.csv`
- `summary.json`
- `input_metadata/`。SharpCap companion file が見つかった場合のみ。
- `input_manifest.json`
- `orientation_scan.csv`
- `orientation_diagnostics.csv`
- `rejection_summary.csv`
- `frame_distribution_summary.csv`
- `spot_assignment_debug.csv`
- `relative_motion_outliers.csv`
- `fwhm_outliers.csv`
- `roi_safety_report.csv`
- `seeing_timeseries.png`
- `r0_timeseries.png`
- `fit_success_rate_timeseries.png`
- `dx_dy_timeseries.png`
- `centroid_scatter.png`
- `fwhm_timeseries.png`
- `peak_timeseries.png`
- `peak_histogram.png`
- `separation_timeseries.png`
- `relative_motion_outliers.png`
- `roi_safety_timeseries.png`
- `orientation_diagnostics.png`
- `valid_vs_rejected_peak_histogram.png`
- `valid_vs_rejected_fwhm_histogram.png`
- `valid_vs_rejected_separation_histogram.png`
- `rejection_summary.png`
- `example_fit_success.png`。利用可能な場合のみ。
- `example_fit_failed.png`。利用可能な場合のみ。

`input_manifest.json` は、その出力フォルダがどのSERから作られたかを記録する
小さな台帳です。`input_ser_path`, `input_ser_filename`, `input_ser_stem`,
`output_root`, `output_dir`, `config_path`, `zenith_deg`, `fps`, `analysis_started_at`,
`ser_metadata`, `copied_companion_files`, `auto_output_name` を確認できます。

`--ser-select all` で複数SERを解析した場合は、指定した出力ディレクトリ直下に
`batch_summary.csv` と `batch_manifest.json` も作成されます。各SERの通常出力は
SERファイル名のサブフォルダに入ります。`batch_summary.csv` には `ser_file`,
`ser_path`, `output_dir`, `summary_json`, `block_results_csv`, `per_frame_fits_csv`,
`total_frames`, `valid_frames`, `frame_fit_success_rate`, `number_of_valid_blocks`,
`median_seeing_zenith_corrected_arcsec`, `median_r0_zenith_m`, `result_reliability`,
`saturated_frame_fraction`, `roi_safety_status`, `orientation_angle_deg` が並び、
SERと出力フォルダの対応を一覧できます。

`batch_manifest.json` には batch 全体の `input_path_original`, `input_was_directory`,
`ser_select_mode`, `number_of_ser_files`, `ser_files`, `output_root`,
`batch_summary_csv`, `entries`, `analysis_started_at`, `analysis_finished_at`, `recursive`
が記録されます。`ser_files` と `entries` には各 SER の処理結果が入ります。

`per_frame_fits.csv` には Gaussian fit の重心、パラメータ、frame validity、reject reason
を出力します。`block_results.csv` には longitudinal/transverse variance、`r0`、観測
seeing、天頂補正後 seeing を出力します。`summary.json` には SER metadata、実際に
使った config、有効ピクセルスケール、時刻ソース、fit 成功率、orientation 推定、
ROI size、入力として指定した元パス、実際に解析したSER、解決後の出力先、警告、
代表 seeing 統計を記録します。

`orientation_scan.csv` は angle ごとの `var_L_px2`, `var_T_px2`, `seeing_L_arcsec`,
`seeing_T_arcsec`, `seeing_mean_arcsec`, `mismatch_abs_log_ratio` を出力します。
`orientation_diagnostics.csv` は同じ内容を診断用の名前で保存したものです。
`rejection_summary.csv` は reject reason の件数、`frame_distribution_summary.csv` は
valid/rejected frame の `separation_px`, FWHM, peak, flux の簡易分布を出力します。
`spot_assignment_debug.csv` は各 frame の候補位置、割り当て後の spot1/spot2、
assignment distance、swap有無を記録します。
`roi_safety_report.csv` は各星像の端距離診断です。`--disable-roi-safety-check` を指定した場合は
`summary.json` に `roi_safety_status: "not_checked"` を記録し、このCSV/PNGは作成しません。

## 開発

テストと lint は以下で実行します。

```bash
pytest
ruff check .
```
