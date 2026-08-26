"""M5 configuration registry; this module does not launch experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PVLDM5Config:
    name: str
    num_prompt_queries: int
    decoder_hidden_size: int = 256
    decoder_layers: int = 2
    decoder_heads: int = 8
    high_resolution_tokens: int = 256

    def parameter_estimate(self, visual_size: int = 1024) -> int:
        # Prompt bank + visual projection + decoder self/cross/FFN weights.
        prompt = self.num_prompt_queries * self.decoder_hidden_size
        projection = visual_size * self.decoder_hidden_size + self.decoder_hidden_size
        attention = self.decoder_layers * 4 * self.decoder_hidden_size**2
        ffn = self.decoder_layers * 2 * self.decoder_hidden_size * (4 * self.decoder_hidden_size)
        return prompt + projection + attention + ffn

    def attention_flops_estimate(self, layout_tokens: int = 256) -> int:
        d = self.decoder_hidden_size
        memory_tokens = self.num_prompt_queries + self.high_resolution_tokens
        return self.decoder_layers * 2 * (
            layout_tokens * layout_tokens * d + layout_tokens * memory_tokens * d
        )

    def as_dict(self) -> dict[str, int | str]:
        payload = asdict(self)
        payload["parameter_estimate"] = self.parameter_estimate()
        payload["attention_flops_estimate"] = self.attention_flops_estimate()
        return payload


M5_CONFIGS = {
    name: PVLDM5Config(name=name, num_prompt_queries=queries)
    for name, queries in (("K16", 16), ("K32", 32), ("K64", 64))
}


def registry() -> dict[str, dict[str, int | str]]:
    return {name: config.as_dict() for name, config in M5_CONFIGS.items()}


if __name__ == "__main__":
    import json

    print(json.dumps(registry(), indent=2, sort_keys=True))
