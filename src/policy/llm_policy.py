import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import logging
from src.policy.base import OneStepPolicy
from src.sokoban import SokobanState


_logger = logging.getLogger(__name__)


_PROMPT = """
You are an expert Sokoban solver.

Rules:
- @ = player
- $ = box
- . = target
- # = wall
- Space = empty

Goal: push all boxes onto targets. If you move into a box, and there is an empty space beyond it, the box is pushed.

Valid actions (choose exactly one):
U = up
D = down
L = left
R = right

Current board:
{state_ascii}

Output exactly one letter from U, D, L, or R.
Do not output anything else.

For example, if the best action is to move left, output "L" without quotes.

Action:"""


class MistralOneStepPolicy(OneStepPolicy):
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str | None = None,
        temperature: float = 1.0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,          # 🔥 critical for Colab
            torch_dtype=torch.float16,
        )

        self.model.eval()
        self.temperature = temperature

        # Action tokens (must be single-token)
        self.action_tokens = {
            "left": self.tokenizer.encode("L", add_special_tokens=False)[0],
            "right": self.tokenizer.encode("R", add_special_tokens=False)[0],
            "up": self.tokenizer.encode("U", add_special_tokens=False)[0],
            "down": self.tokenizer.encode("D", add_special_tokens=False)[0],
        }

        for k, v in self.action_tokens.items():
            decoded = self.tokenizer.decode([v])
            _logger.info("Action %s → token '%s' (id=%d)", k, decoded, v)

        _logger.debug(f"Token id for left: {self.action_tokens['left']}")
        _logger.debug(f"Token id for right: {self.action_tokens['right']}")
        _logger.debug(f"Token id for up: {self.action_tokens['up']}")
        _logger.debug(f"Token id for down: {self.action_tokens['down']}")

        _logger.info("Loaded Mistral policy: %s", model_name)

    @torch.no_grad()
    def predict(self, state: SokobanState) -> list[tuple[str, float]]:
        prompt = _PROMPT.format(state_ascii=state.render())
        _logger.debug("LLM prompt:\n%s", prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]  # 👈 next-token logits

        _logger.debug("Decision logits size: %d", logits.size(0))

        _logger.debug("Top 10 most likely tokens:")
        top_ten_most_likely_token_ids = torch.topk(logits, k=10).indices.tolist()
        for token_id in top_ten_most_likely_token_ids:
            token_str = self.tokenizer.decode([token_id])
            token_logit = logits[token_id].item()
            _logger.debug("  Token '%s' (id %d): logit %.3f", token_str, token_id, token_logit)

        # Extract action logits
        action_logits = torch.tensor(
            [logits[token_id] for token_id in self.action_tokens.values()],
            device=logits.device,
        )

        _logger.debug("Raw action logits: %s", action_logits.tolist())

        # Temperature + softmax
        action_logits /= self.temperature
        probs = F.softmax(action_logits, dim=0)

        results = list(zip(self.action_tokens.keys(), probs.tolist()))

        _logger.debug(
            "LLM action distribution: %s",
            {a: f"{p:.3f}" for a, p in results},
        )

        return results


# class LLMOneStepPolicy(OneStepPolicy):
#     def __init__(
#         self,
#         model_name: str = "google/flan-t5-small",
#         device: str | None = None,
#         temperature: float = 1.0,
#     ):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()

#         self.temperature = temperature

#         # Token IDs for actions (single-token for T5)
#         self.action_tokens = {
#             "up": self.tokenizer.encode("A", add_special_tokens=False)[0],
#             "down": self.tokenizer.encode("B", add_special_tokens=False)[0],
#             "left": self.tokenizer.encode("C", add_special_tokens=False)[0],
#             "right": self.tokenizer.encode("D", add_special_tokens=False)[0],
#         }

#         _logger.info("Loaded LLM policy: %s on %s", model_name, self.device)

#     @torch.no_grad()
#     def predict(self, state: SokobanState) -> list[tuple[str, float]]:
#         prompt = sokoban_prompt(state.render())
#         _logger.debug("LLM prompt: %s", prompt)

#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#         ).to(self.device)

#         # Forward pass — get logits for first output token
#         # outputs = self.model(
#         #     **inputs,
#         #     decoder_input_ids=torch.tensor(
#         #         [[self.model.config.decoder_start_token_id]],
#         #         device=self.device,
#         #     ),
#         # )
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=1,
#             return_dict_in_generate=True,
#             output_scores=True,
#             do_sample=False,  # greedy is fine
#         )
#         # scores[0] = logits at the first generated step
#         decision_logits = outputs.scores[0][0]  # shape: [vocab_size]
#         _logger.debug("LLM decision logits size: %d", decision_logits.size(0))

#         # Actual most likely tokens
#         _logger.debug("Top 10 most likely tokens:")
#         top_ten_most_likely_token_ids = torch.topk(decision_logits, k=10).indices.tolist()
#         for token_id in top_ten_most_likely_token_ids:
#             token_str = self.tokenizer.decode([token_id])
#             token_logit = decision_logits[token_id].item()
#             _logger.debug("  Token '%s' (id %d): logit %.3f", token_str, token_id, token_logit)

#         action_logits = torch.tensor(
#             [decision_logits[token_id] for token_id in self.action_tokens.values()],
#             device=self.device,
#         )
#         _logger.debug("Raw action logits: %s", action_logits.tolist())

#         #check
#         generated_token_id = outputs.sequences[0, -1].item()
#         generated_token = self.tokenizer.decode([generated_token_id])
#         _logger.debug("Generated token: %s", generated_token)

#         # Temperature scaling
#         action_logits /= self.temperature

#         probs = F.softmax(action_logits, dim=0)

#         results = list(zip(self.action_tokens.keys(), probs.tolist()))

#         _logger.debug(
#             "LLM action distribution: %s",
#             {a: f"{p:.2f}" for a, p in results},
#         )

#         return results
