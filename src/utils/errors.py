class PipelineError(Exception):
    """Base class for all expected, user-facing failures."""


class ConfigError(PipelineError):
    pass


class VideoInputError(PipelineError):
    pass


class FFmpegError(PipelineError):
    pass


class TranscriptionError(PipelineError):
    pass


class LLMError(PipelineError):
    pass