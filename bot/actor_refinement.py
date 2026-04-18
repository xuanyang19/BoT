import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from .evaluate import evaluate_correctness

# =========================
# JSON parsing (no PCRE recursion)
# =========================
def _strip_code_fences(s: str) -> str:
    s = str(s).strip()
    if s.startswith("```"):
        s = s.split("```", 2)[-1].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    return s


def _best_effort_json(s: str) -> Optional[dict]:
    s = _strip_code_fences(s or "")
    try:
        return json.loads(s)
    except Exception:
        pass
    # manual brace matching
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(s[start:], start=start):
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(s[start:i+1])
                        except Exception:
                            break
        start = s.find("{", start+1)
    return None

def _gemini_strip_code_fences(s: str) -> str:
    """
    Convert ```json ... ``` block into a valid JSON-like string.
    If the inner content doesn't start with '{' or '[',
    it wraps it with '{' + content + '}'.
    """
    if not s:
        return ""

    s = s.strip()

    # match only ```json fenced blocks
    match = re.match(r"^```json\s*\n([\s\S]*?)```$", s)
    if match:
        inner = match.group(1).strip()
        # If content isn't a full object/list, wrap it with braces
        if not (inner.startswith("{") or inner.startswith("[")):
            # handle `"response": [...]` case
            inner = "{\n" + inner + "\n}"
        return inner
    return s

def _gemini_best_effort_json(s: str) -> Optional[dict]:
    s = _gemini_strip_code_fences(s or "")
    try:
        return json.loads(s)
    except Exception:
        pass
    # manual brace matching
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(s[start:], start=start):
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(s[start:i+1])
                        except Exception:
                            break
        start = s.find("{", start+1)
    return None

# =========================
# Prompts (dataset-aware)
# =========================
ACTION_PROMPT_MEDQA = """
Please answer the following medical multiple choice question as accurately as possible. Define any key premises and ensure consistency with clinical facts.
Think step-by-step and provide clear reasoning.
Output a valid JSON object:
{"answer": "A", "reasoning": "...", "confidence_score": 0.0}
""".strip()

ACTION_PROMPT_WINOGRANDE = """
You are given an incomplete sentence with a blank. Your task is to choose which option (1 or 2) best completes the sentence in a way that makes sense.


The sentence will have two options that could fill in the blank. Choose the option that creates the most logical and coherent sentence based on common sense reasoning and real-world knowledge.


Think step-by-step about:
- Which option makes the sentence grammatically correct
- Which option creates a logical and sensible meaning
- Real-world knowledge about the entities mentioned


Output a valid JSON object with your answer as 1 or 2:
{"answer": "1", "reasoning": "...", "confidence_score": 0.0}
""".strip()


REFLECTION_SYSTEM_PROMPT = """
You are a reflection agent to help refine the answers. Here are <<N>> questions, each with the previous model's answer.
For each, critique the model answer for accuracy, completeness, and reasoning, comparing across all answers and their reasoning paths in the batch to identify areas for improvement and give a peer confidence score to quantify how possible the answer is correct.
Make sure you understand each question-answer pair and give detailed explanations to them, Carefully decide if a reevaluation is needed for each case.
For each, provide: (1) whether to trigger reevaluation (true/false) and improve answer, (2) summary assessment, (3) peer confidence score for the current answer(0.0-1.0), (4) suggestions for improvement(empty if reevaluation is false).
Output a JSON list, one entry per question, strictly in format:
"response:[{trigger_reevaluation: bool, summary_comment: str, confidence_score: float(0.0-1.0), suggestions: str}]"
"""

def build_reflection_prompt(n: int) -> str:
    return REFLECTION_SYSTEM_PROMPT.replace("<<N>>", str(n))


# =========================
# Conversation helpers
# =========================
def run_conversation_actor(client, prompt, query, model, tools, tool_mapping, max_iterations=5):
    """
    Run a conversation with the LLM using tools as needed.
    Returns: conversation_history, tool_call_count
    """

    RESPONSE_FORMAT = {"type": "json_object"}

    conversation_history: List[Dict[str, Any]] = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': query},
    ]
    iteration_count = 0
    tool_call_count = 0
    while iteration_count < max_iterations:
        try:
            if 'gemini' not in model:
                RESPONSE_FORMAT = {"type": "json_object"}
                res = client.chat.completions.create(
                    model=model,
                    messages=conversation_history,
                    response_format=RESPONSE_FORMAT,
                    tools=tools,
                    temperature=0.0,
                    max_tokens=4096,
                )
            else:
                res = client.chat.completions.create(
                    model=model,
                    messages=conversation_history,
                    tools=tools,
                    temperature=0.0,
                    max_tokens=4096,
                )
            # harden SDK object to pure dict
            res_dict = json.loads(res.model_dump_json())
            assistant_msg = res_dict['choices'][0]['message']
            finish = res_dict['choices'][0].get('finish_reason', '')
            conversation_history.append(assistant_msg)

            # No tool calls → done
            if finish != 'tool_calls':
                break

            # Handle tool calls
            for tool_call in assistant_msg.get('tool_calls', []) or []:
                tool_call_count += 1
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
                try:
                    tool_fn = tool_mapping.get(tool_name)
                    tool_response = str(tool_fn(**tool_args)) if tool_fn else f"Unknown tool: {tool_name}"
                except Exception as e:
                    tool_response = f"Error executing tool: {str(e)}"
                conversation_history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call['id'],
                    'content': tool_response
                })
            iteration_count += 1

        except Exception as e:
            conversation_history.append({'role': 'system', 'content': f"An error occurred: {str(e)}"})
            break

    return conversation_history, tool_call_count


def run_conversation_ref(client, prompt, query, model):
    """
    Reflection call: no tools. Expects JSON in content and returns parsed dict.
    """
    RESPONSE_FORMAT = {"type": "json_object"}

    conversation_history = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': query},
    ]
    try:
        if 'gemini' not in model:
            res = client.chat.completions.create(
                model=model,
                messages=conversation_history,
                response_format=RESPONSE_FORMAT,
                temperature=0.0,
                max_tokens=4096,
            )
            content = res.choices[0].message.content
        else:
            res = client.chat.completions.create(
                model=model,
                messages=conversation_history,
                temperature=0.0,
                max_tokens=4096,
            )
            content = res.choices[0].message.content
        if 'gemini' in model:
            output = _gemini_best_effort_json(content)
        else:
            output = _best_effort_json(content)
        return output or {"response": []}
    except Exception as e:
        return {"response": [], "error": str(e)}


# =========================
# Query builders
# =========================
def _letters(n: int) -> List[str]:
    return [chr(ord('A') + i) for i in range(max(0, min(n, 26)))]


def make_query(row: pd.Series, dataset: str) -> Tuple[str, str]:
    """Build query and prompt for supported datasets."""

    if dataset == "medqa":
        opts = row.get("options", None) or []
        Ls = _letters(len(opts) if opts else 4)
        opt_txt = "" if not opts else ("\nOptions:\n" + "\n".join(f"{L}. {opt}" for L, opt in zip(Ls, opts)))
        user = f"Question: {row['question']}{opt_txt}"
        return user, ACTION_PROMPT_MEDQA

    if dataset == "winogrande":
        question = row.get("question", "")
        opts = row.get("options", [])
        if len(opts) == 2:
            user = f"Incomplete sentence: {question} _____\n\nOptions:\n1. {opts[0]}\n2. {opts[1]}\n\nWhich option (1 or 2) best completes the sentence?"
        else:
            user = f"Incomplete sentence: {question}"
        return user, ACTION_PROMPT_WINOGRANDE

    return None, None
