from prime import llm

MODEL = "z-ai/glm-5.3-flash"
REASONING = "low"
INCLUDE_REASONING = False

response, cost = llm(
    MODEL,
    [{"role": "user", "content": "What is Prime Intellect?"}],
    reasoning=REASONING,
    include_reasoning=INCLUDE_REASONING,
)

print(response)
print(f"Cost: ${cost:.6f}")
