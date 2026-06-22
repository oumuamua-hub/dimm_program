# DIMM Analyzer Readability Refactor Design

## Objective

`dimm-analyzer` の処理結果と外部契約を変えずに、主要な処理を人間が追いやすい責務へ分割する。
レビュー担当者が、フレーム処理、診断集計、summary 作成、CLI 実行制御を別々に読める状態を完成条件とする。

## Non-Behavioral Contract

この作業では次の項目を変更しない。

- `dimm_analyzer.pipeline.analyze_ser` と `analyze_frame_sequence` の公開 import path と引数
- Typer のコマンド名、オプション名、既定値、help、終了コード
- `dimm_analyzer.cli.analyze_ser` と `select_analysis_paths` を差し替えられるテスト境界
- フレームの検出、fit、reject、tracking、orientation、ROI safety の判定順
- 飽和コアを `saturated`、ROI 内の孤立高輝度画素を `hot_pixel_or_roi_outlier` とする分類
- `FitResult`、`FrameResult`、`BlockResult`、`AnalysisResult` のフィールドと既定値
- CSV のファイル名、列名、列順、値の表現
- JSON と manifest のキー、値、null の扱い
- 例外型、利用者向け例外文言、warning 文言
- plot の種類、ファイル名、元データ

## Architecture

### `pipeline.py`

公開 facade と解析順序を担当する。SER の読み込み、フレーム列の解析、各処理ステージの呼び出し、
`AnalysisResult` の組み立て、出力呼び出しを残す。内部実装を移動した既存関数は `pipeline.py` で
import/re-export し、既存の module-level symbol を維持する。

### `_frame_processing.py`

1フレームを結果へ変換する責務を持つ。

- source detection と spot pair 選択
- ROI 抽出と Gaussian fit
- spot identity の割り当て
- center jump と tracking の判定
- `FrameResult` への座標・診断値の設定

処理順を変えないため、既存関数をまとまり単位で移動し、条件式は書き換えない。

### `_diagnostics.py`

解析済みの `FrameResult` から診断行を作る純粋な後処理を担当する。

- FWHM outlier filter
- relative motion filter
- spot assignment debug rows
- ROI safety points
- orientation scan rows
- rejection summary と frame distribution rows

filter が `FrameResult` を更新する順序は、現在の `analyze_frame_sequence` と同じに保つ。

### `_summary.py`

最終 summary と整合性検証を担当する。

- saturation、valid/rejected、seeing、r0 の集計
- quality flags と result reliability
- `summary.json` 用 dictionary の構築
- summary と frame/block rows の整合性検証
- summary のみに使う統計補助関数

summary dictionary は既存と同じ挿入順で構築する。JSON のキー順まで前後比較の対象にする。

### CLI Support

`cli.py` には Typer command signature と top-level routing を残す。既存テストの monkeypatch 境界を
壊さない範囲で、出力先解決、manifest row 構築、batch/comparison 用の純粋な整形処理を
`_cli_support.py` へ移す。実行時依存を持つ処理を移す場合は、`cli.py` から関数として明示的に渡し、
module-level monkeypatch が引き続き反映される形にする。

## Data Flow

1. `cli.py` が入力、設定、出力先を解決する。
2. `pipeline.analyze_ser` が SER metadata、dark、時刻、ROI safety の事前情報を準備する。
3. `pipeline.analyze_frame_sequence` が `_frame_processing` をフレームごとに呼ぶ。
4. `_diagnostics` を既存順序で呼び、frame validity と診断 rows を確定する。
5. orientation と block results を現在と同じ順序で計算する。
6. `_summary` が summary を構築し、表データとの整合性を検証する。
7. `pipeline.write_outputs` が既存の report/plot writer を呼ぶ。

## Comments And Naming

コメントは「何をしているか」ではなく、コードだけでは分かりにくい理由と不変条件を説明する。

- reject/filter の順序が結果へ影響する箇所
- spot identity を前フレームから維持する規則
- saturation を core-first で判定する理由
- robust percentile を ROI 推奨値に使う理由
- summary と CSV の source of truth の関係
- Typer/monkeypatch 互換性のため facade に残す処理

単純な代入や関数名を言い換えるコメントは追加しない。既存の日本語利用者向け文言は変更しないが、
開発者向け docstring とコメントは既存コードに合わせて簡潔な英語を基本とする。

## Error Handling

例外を捕捉する場所と順序は変えない。移動した内部関数から同じ例外をそのまま伝播させる。
`FitResult.failed()` へ格納する failure reason、CLI が `DimmAnalyzerError` を `typer.Exit(1)` へ
変換する処理、予期しない例外の表示形式を維持する。

## Verification Strategy

実装前に現行コードで次の baseline を取得する。

- 全 pytest の結果
- 通常の Ruff 結果
- deterministic synthetic pipeline の CSV/JSON 出力
- CLI runner tests が確認する option routing、batch、comparison、manifest の挙動

各責務を移動するたびに関連テストを実行し、最後に次を確認する。

1. `python3 -m pytest` が全件成功する。
2. `python3 -m ruff check .` が成功する。
3. synthetic pipeline の全 CSV と JSON が baseline と一致する。
4. plot はファイル一覧、画像サイズ、pixel array が baseline と一致する。
5. 公開 import path と関数 signature が baseline と一致する。
6. README に記載された CLI option と出力ファイルが変更されていない。
7. `FitResult.prefixed_dict()`、`PER_FRAME_COLUMNS`、summary/manifest keys が変更されていない。

## Scope Limits

- アルゴリズム改善、性能改善、設定値変更は行わない。
- Pydantic model や result schema の再設計は行わない。
- Python 対応バージョンや依存パッケージを変更しない。
- README の機能説明は、コード契約に差分がない限り変更しない。
- 可読性に直接寄与しない formatting や命名変更は行わない。
