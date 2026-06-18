"""dimm-analyzer 用の例外クラス。"""


class DimmAnalyzerError(Exception):
    """パッケージ内エラーの基底例外。"""


class ConfigError(DimmAnalyzerError):
    """設定が不正または不足している場合の例外。"""


class SERFormatError(DimmAnalyzerError):
    """SER ファイルを解析できない、または未対応の場合の例外。"""


class DetectionError(DimmAnalyzerError):
    """スポット検出に失敗した場合の例外。"""


class FitError(DimmAnalyzerError):
    """Gaussian fitting が予期せず失敗した場合の例外。"""


class PipelineError(DimmAnalyzerError):
    """解析パイプラインを継続できない場合の例外。"""
