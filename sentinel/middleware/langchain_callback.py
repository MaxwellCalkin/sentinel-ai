"""LangChain integration for Sentinel AI safety guardrails.

Provides a callback handler that scans LLM inputs/outputs in real-time,
and a chain wrapper for inline safety checks.

Usage with callback:
    from sentinel.middleware.langchain_callback import SentinelCallbackHandler

    handler = SentinelCallbackHandler()
    llm = ChatOpenAI(callbacks=[handler])
    response = llm.invoke("Hello!")
    # Check handler.findings for any safety issues

Usage with chain wrapper:
    from sentinel.middleware.langchain_callback import SentinelChain

    chain = SentinelChain(llm=your_llm)
    result = chain.invoke({"input": "Hello!"})
    # result["blocked"] is True if safety scan failed
"""

from __future__ import annotations

from typing import Any, Sequence
from sentinel.core import SentinelGuard, ScanResult, Finding, RiskLevel


class SentinelCallbackHandler:
    """LangChain callback handler that scans LLM I/O with Sentinel.

    Compatible with LangChain's BaseCallbackHandler interface without
    requiring langchain as a dependency. Works with both legacy and
    LCEL chains.

    Attributes:
        findings: List of all findings from scans during the chain run.
        blocked: Whether any input or output was blocked.
        scans: List of all ScanResult objects.
    """

    name = "SentinelCallbackHandler"

    def __init__(
        self,
        guard: SentinelGuard | None = None,
        scan_input: bool = True,
        scan_output: bool = True,
        raise_on_block: bool = False,
    ):
        self._guard = guard or SentinelGuard.default()
        self._scan_input = scan_input
        self._scan_output = scan_output
        self._raise_on_block = raise_on_block
        self.findings: list[Finding] = []
        self.scans: list[ScanResult] = []
        self.blocked: bool = False

    def reset(self) -> None:
        """Clear findings between runs."""
        self.findings.clear()
        self.scans.clear()
        self.blocked = False

    def _handle_scan(self, scan: ScanResult, source: str) -> None:
        self.scans.append(scan)
        self.findings.extend(scan.findings)
        if scan.blocked:
            self.blocked = True
            if self._raise_on_block:
                raise SentinelBlockedError(
                    f"{source} blocked by Sentinel: "
                    f"risk={scan.risk.value}, "
                    f"findings={len(scan.findings)}"
                )

    # --- LangChain callback interface methods ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Scan prompts before they reach the LLM."""
        if not self._scan_input:
            return
        for prompt in prompts:
            scan = self._guard.scan(prompt)
            self._handle_scan(scan, "LLM input")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        """Scan chat messages before they reach the model."""
        if not self._scan_input:
            return
        for message_batch in messages:
            for message in message_batch:
                text = self._extract_message_text(message)
                if text:
                    scan = self._guard.scan(text)
                    self._handle_scan(scan, "Chat input")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Scan LLM output after generation."""
        if not self._scan_output:
            return
        text = self._extract_response_text(response)
        if text:
            scan = self._guard.scan(text)
            self._handle_scan(scan, "LLM output")

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        pass

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        """Extract text from a LangChain message object or dict."""
        if isinstance(message, dict):
            return message.get("content", "")
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
        return ""

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text from a LangChain LLMResult or similar."""
        if hasattr(response, "generations"):
            texts = []
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "text"):
                        texts.append(gen.text)
                    elif hasattr(gen, "message") and hasattr(gen.message, "content"):
                        texts.append(gen.message.content)
            return " ".join(texts)
        return ""


class SentinelBlockedError(Exception):
    """Raised when Sentinel blocks content and raise_on_block=True."""
    pass


def create_sentinel_callback(
    block_threshold: RiskLevel = RiskLevel.HIGH,
    scan_input: bool = True,
    scan_output: bool = True,
    raise_on_block: bool = False,
) -> SentinelCallbackHandler:
    """Factory function to create a configured Sentinel callback handler."""
    guard = SentinelGuard.default()
    guard._block_threshold = block_threshold
    return SentinelCallbackHandler(
        guard=guard,
        scan_input=scan_input,
        scan_output=scan_output,
        raise_on_block=raise_on_block,
    )
