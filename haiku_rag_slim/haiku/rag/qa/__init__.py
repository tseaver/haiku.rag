from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.qa.agent import QuestionAnswerAgent
from haiku.rag.qa.prompts import QA_SYSTEM_PROMPT, QA_SYSTEM_PROMPT_WITH_RAPTOR


async def get_qa_agent(
    client: HaikuRAG,
    config: AppConfig = Config,
    system_prompt: str | None = None,
) -> QuestionAnswerAgent:
    """Factory function to get a QA agent based on the configuration.

    Args:
        client: HaikuRAG client instance.
        config: Configuration to use. Defaults to global Config.
        system_prompt: Optional custom system prompt. If not provided,
            uses appropriate prompt based on RAPTOR availability.

    Returns:
        A configured QuestionAnswerAgent instance.
    """
    if system_prompt is None:
        has_raptor = await client.raptor_repository.has_nodes()
        system_prompt = QA_SYSTEM_PROMPT_WITH_RAPTOR if has_raptor else QA_SYSTEM_PROMPT

    return QuestionAnswerAgent(
        client=client,
        model_config=config.qa.model,
        config=config,
        system_prompt=system_prompt,
    )
