DOC_LOAD_ERROR_MESSAGE = r"知识库文档加载失败：未找到指定知识库目录或文件。请检查 D:\cd\docs 目录下是否存在 Word/TXT/PDF 文档。"
LLM_ERROR_MESSAGE = r"大模型接口暂时不可用：请检查 API Key、Base URL、模型名称配置，或确认当前网络可访问模型 API 服务。"
NO_MATCH_MESSAGE = "暂无相关知识库依据，无法为您解答"


class DocumentLoadError(Exception):
    pass


class LLMServiceError(Exception):
    pass


class RetrievalError(Exception):
    pass
