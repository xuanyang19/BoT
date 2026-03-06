# -*- coding: utf-8 -*-
"""
Main pipeline runner that orchestrates the entire actor refinement process.
"""

import os
import json
import sys
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Import our generic API client
from api_client import get_openai_client
from model import SUPPORTED_MODELS, DEFAULT_MODEL

# Import our modular components
from dataloader import load_dataset_router
from batching import build_batches
from actor_refinement import (
    run_conversation_actor, run_conversation_ref, make_query,
    evaluate_correctness, build_reflection_prompt, _best_effort_json, _gemini_best_effort_json
)


# =========================
# METRIC HELPERS
# =========================
def flatten_bool(x):
    """Convert various boolean representations to True/False/None."""
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    return None


def basic_acc_safe(correctness):
    """Calculate various accuracy metrics from correctness data."""
    final_hits = final_cnt = orig_hits = orig_cnt = best_hits = best_cnt = 0
    for att in correctness:
        if not att:
            continue
        # Original
        o = flatten_bool(att[0])
        if o is not None:
            orig_cnt += 1
            orig_hits += int(o)

        # Final
        f = flatten_bool(att[-1])
        if f is not None:
            final_cnt += 1
            final_hits += int(f)

        # Best (include all steps including final)
        vals = [flatten_bool(v) for v in att if flatten_bool(v) is not None]
        if vals:
            best_cnt += 1
            best_hits += int(any(vals))
        elif f is not None:   # <-- NEW: count items that only have a final step
            best_cnt += 1
            best_hits += int(f)

    return {
        "original_acc": orig_hits / orig_cnt if orig_cnt else float("nan"),
        "final_acc": final_hits / final_cnt if final_cnt else float("nan"),
        "best_acc": best_hits / best_cnt if best_cnt else float("nan"),
        "N_items": len(correctness),
        "orig_den": orig_cnt, "final_den": final_cnt, "best_den": best_cnt
    }


class ActorRefinementPipeline:
    def __init__(self, dataset=None, split="test", batch_size=8, embed_cache=None,
                 batching="sequential", use_tools=False, out_dir="outputs", limit=None, seed=42,
                 model=None):
        # Configuration
        self.dataset = dataset
        if self.dataset is None or len(self.dataset) == 0:
            raise Exception("No dataset given!")
        self.split = split
        self.batch_size = batch_size
        self.embed_cache = embed_cache
        self.batching = batching
        self.use_tools = use_tools
        self.out_dir = out_dir
        self.limit = limit
        self.seed = seed
        self.model = model or DEFAULT_MODEL

        # Client setup - uses generic OpenAI client
        # Set OPENAI_API_KEY environment variable before running
        self.client = get_openai_client()

        # Tools setup - disabled by default for open-source version
        # Users can add their own tools here
        self.ALL_TOOLS = []
        self.tool_mapping = {}
        self.tools = self.ALL_TOOLS if use_tools else []
        self.MAX_REFLECTIONS = 8

        # Output setup
        self.RUN_NAME = f"{dataset}_{split}_reflect"
        self.TIMESTAMP = datetime.now().strftime("%m%d_%H%M%S")
        self.RUN_DIR = Path(out_dir) / f"{self.RUN_NAME}_{batch_size}_{self.TIMESTAMP}"
        self.RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.CKPT_BASENAME = "ckpt"

        # Results storage
        self.correctness: List[List[Optional[bool]]] = []
        self.results: List[List[Dict[str, Any]]] = []
        self.reflections: List[List[Dict[str, Any]]] = []
        self.confidences: List[List[Optional[float]]] = []
        self.ref_confidences: List[List[Optional[float]]] = []
        self.tool_calls: List[List[int]] = []

        # History and timing storage
        self.conversation_histories: List[List[List[Dict[str, Any]]]] = []
        self.reflection_histories: List[List[Dict[str, Any]]] = []
        self.processing_times: List[float] = []

        # Checkpoint tracking
        self._processed_count = 0
        self._total = 0
        self._pct_marks = [20, 40, 60, 80]
        self._ckpt_targets: List[tuple[int,int]] = []

    def load_data(self):
        """Load and prepare the dataset."""
        print(f"Loading dataset: {self.dataset}/{self.split}")
        df = load_dataset_router(self.dataset, self.split)
        print(df.head())
        print(f"Loaded dataset with {len(df)} rows...")

        # Apply limit if specified
        if self.limit is not None and self.limit > 0 and len(df) > self.limit:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(df), size=self.limit, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
            print(f"[subset] LIMIT={self.limit}, seed={self.seed}; N={len(df)}")

        if len(df) == 0:
            raise RuntimeError(f"Loaded 0 examples for {self.dataset}/{self.split}.")

        self.df = df
        self._total = len(df)
        self._setup_checkpoints()
        return df

    def _setup_checkpoints(self):
        """Setup checkpoint targets."""
        _seen = set()
        for p in self._pct_marks:
            thresh = max(1, math.ceil(self._total * p / 100.0))
            if thresh not in _seen:
                self._ckpt_targets.append((p, thresh))
                _seen.add(thresh)
        self._ckpt_targets.sort(key=lambda x: x[1])

    def prepare_batches(self):
        """Create batches using the specified strategy."""
        print(f"Building batches with strategy: {self.batching}")
        batches = build_batches(self.df, self.batch_size, self.embed_cache, self.batching)

        print(f"[debug] dataset={self.dataset}/{self.split}, N={len(self.df)}, BATCH_SIZE={self.batch_size}")
        print(f"[batching] strategy={self.batching}, num_batches={len(batches)}")

        if self._ckpt_targets:
            plan = ", ".join([f"{p}%@{c}" for p,c in self._ckpt_targets])
            print(f"[checkpoint plan] will save at: {plan}")

        return batches

    def _to_py(self, o):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(o, dict):
            return {k: self._to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self._to_py(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    def _atomic_json_dump(self, data, path: Path):
        """Atomic JSON dump to avoid corruption."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def _build_payload(self, extra_meta=None):
        """Build checkpoint payload."""
        # Calculate accuracy metrics
        accuracy_metrics = basic_acc_safe(self.correctness)

        # Calculate tool call statistics
        tool_call_stats = self._calculate_tool_call_stats()

        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": self.dataset, "split": self.split,
            "batch_size": self.batch_size, "max_reflections": self.MAX_REFLECTIONS,
            "model": self.model, "batching_strategy": self.batching,
            "embed_model": "intfloat/e5-mistral-7b-instruct" if self.batching=="kmeans" else None,
            "use_tools": self.use_tools,
            "out_dir": str(self.RUN_DIR),
        }
        # Add accuracy metrics to meta
        meta.update(accuracy_metrics)
        # Add tool call statistics to meta
        meta.update(tool_call_stats)
        if extra_meta:
            meta.update(extra_meta)
        return self._to_py({
            "meta": meta,
            "correctness": self.correctness,
            "results": self.results,
            "reflections": self.reflections,
            "confidences": self.confidences,
            "ref_confidences": self.ref_confidences,
            "tool_calls": self.tool_calls,
            "processing_times": self.processing_times,
        })

    def _calculate_tool_call_stats(self):
        """Calculate tool call statistics."""
        if not self.tool_calls:
            return {"total_tool_calls": 0, "avg_tool_calls_per_item": 0.0, "tool_usage_rate": 0.0}

        all_calls = []
        items_with_tools = 0

        for item_calls in self.tool_calls:
            total_calls = sum(item_calls) if item_calls else 0
            all_calls.append(total_calls)
            if total_calls > 0:
                items_with_tools += 1

        total_calls = sum(all_calls)
        avg_calls = total_calls / len(all_calls) if all_calls else 0
        usage_rate = items_with_tools / len(all_calls) if all_calls else 0

        return {
            "total_tool_calls": total_calls,
            "avg_tool_calls_per_item": avg_calls,
            "tool_usage_rate": usage_rate,
            "items_with_tool_usage": items_with_tools,
            "total_items": len(all_calls)
        }

    def _save_histories_incrementally(self, new_conversations, new_reflections, start_idx):
        """Save histories incrementally to prevent memory crashes."""
        # Save conversation histories to single file
        if new_conversations:
            conv_path = self.RUN_DIR / "conversations.jsonl"
            try:
                with open(conv_path, 'a', encoding='utf-8') as f:
                    for i, conv_hist in enumerate(new_conversations):
                        entry = {
                            "index": start_idx + i,
                            "conversation_history": self._to_py(conv_hist)
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Warning: Failed to save conversation histories: {e}")

        # Save reflection histories to single file
        if new_reflections:
            ref_path = self.RUN_DIR / "reflections.jsonl"
            try:
                with open(ref_path, 'a', encoding='utf-8') as f:
                    for i, ref_hist in enumerate(new_reflections):
                        entry = {
                            "index": start_idx + i,
                            "reflection_history": self._to_py(ref_hist)
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Warning: Failed to save reflection histories: {e}")

    def save_checkpoint(self, tag: str):
        """Save checkpoint."""
        # Save main results
        out_path = self.RUN_DIR / f"{self.CKPT_BASENAME}_{tag}.json"
        self._atomic_json_dump(
            self._build_payload({"tag": tag, "processed_count": self._processed_count}),
            out_path
        )
        print(f"[checkpoint] saved -> {out_path}")

    def _maybe_checkpoint(self):
        """Check if checkpoint should be saved."""
        while self._ckpt_targets and self._processed_count >= self._ckpt_targets[0][1]:
            pct, cnt = self._ckpt_targets.pop(0)
            self.save_checkpoint(f"{pct}pct_{cnt}")

    def process_batch(self, batch_indices, b_idx):
        """Process a single batch through actor and reflection phases."""
        import time
        batch_start_time = time.time()
        batch_df = self.df.iloc[batch_indices]
        print(f"Processing batch {b_idx}: indices {batch_indices} (size={len(batch_df)})")

        batch_correct = [[] for _ in range(len(batch_df))]
        batch_results = [[] for _ in range(len(batch_df))]
        batch_refs = [[] for _ in range(len(batch_df))]
        batch_conf = [[] for _ in range(len(batch_df))]
        batch_ref_conf = [[] for _ in range(len(batch_df))]
        batch_tool_calls = [[] for _ in range(len(batch_df))]
        batch_queries, gold_rows = [], []
        batch_conversation_histories = [[] for _ in range(len(batch_df))]
        batch_reflection_histories = [[] for _ in range(len(batch_df))]
        batch_times = []

        # Actor pass
        for i, (_, row) in enumerate(batch_df.iterrows()):
            item_start_time = time.time()
            user_query, system_prompt = make_query(row, self.dataset)
            result_msgs, tool_count = run_conversation_actor(
                self.client, system_prompt, user_query, self.model,
                self.tools, self.tool_mapping
            )

            # Parse final assistant JSON
            try:
                if not result_msgs:
                    raise RuntimeError("Empty conversation history from actor call")
                last_msg = result_msgs[-1]
                raw_content = last_msg.get('content', '') if isinstance(last_msg, dict) else ''
                # for gemini
                if '```json' in raw_content:
                    response = _gemini_best_effort_json(raw_content) or {}
                else:
                    response = _best_effort_json(raw_content) or {}
                conf = response.get('confidence_score')
                ok = evaluate_correctness(self.dataset, row, response)
            except Exception as e:
                print(f"Warning: failed to parse output for row {i}: {e}")
                response, ok, conf = {}, None, None

            batch_results[i].append(response)
            batch_correct[i].append(ok)
            batch_conf[i].append(conf)
            batch_tool_calls[i].append(tool_count)
            batch_queries.append(user_query)
            gold_rows.append(row.to_dict())
            # Store conversation history and timing
            batch_conversation_histories[i].append(result_msgs)
            item_time = time.time() - item_start_time
            batch_times.append(item_time)

        # Reflection loop
        for r in range(self.MAX_REFLECTIONS):
            batch_for_reflection, idxs = [], []
            for i in range(len(batch_queries)):
                if (r == 0) or (batch_refs[i] and batch_refs[i][-1].get('trigger_reevaluation', False)):
                    batch_for_reflection.append({
                        "question": batch_queries[i],
                        "model_answer": batch_results[i][-1]
                    })
                    idxs.append(i)

            if not batch_for_reflection:
                break

            reflection_input = json.dumps(batch_for_reflection, ensure_ascii=False)
            reflection_prompt = build_reflection_prompt(len(batch_for_reflection))
            reflection_result = run_conversation_ref(self.client, reflection_prompt, reflection_input, self.model)
            #    print("-----------------------------reflection_result-----------------------------")
            #    print(reflection_result)
            try:
                # Case 1: assume dict with key "response"
                feedback_list = reflection_result.get("response", [])
            except Exception as e:
                try:
                    # Case 3: extract substring between [ and ]
                    s = str(reflection_result)
                    start = s.find("[")
                    end = s.rfind("]")
                    if start != -1 and end != -1 and start < end:
                        chunk = s[start:end+1]
                        try:
                            import ast
                            feedback_list = ast.literal_eval(chunk)  # Python: True/False/None
                        except Exception:
                            # Case 4: if not Python but JSON format
                            try:
                                feedback_list = json.loads(chunk)          # JSON: true/false/null
                            except Exception:
                                feedback_list = []
                    else:
                        feedback_list = []
                except Exception as e2:
                    print("Error:", str(e), str(e2))
                    feedback_list = []

            print(f"Parsed {len(feedback_list)} reflection feedbacks")
            #    print(feedback_list)
            # Store reflection history for this round
            reflection_round_data = {
                'round': r + 1,
                'reflection_input': reflection_input,
                'reflection_result': reflection_result,
                'feedback_list': feedback_list
            }
            # Store reflection data for items that participated in this round
            for orig_i in idxs:
                if orig_i < len(batch_reflection_histories):
                    batch_reflection_histories[orig_i].append(reflection_round_data)

            any_triggered = False
            for j, orig_i in enumerate(idxs):
                feedback = feedback_list[j] if j < len(feedback_list) and isinstance(feedback_list[j], dict) else {}
                ref_conf = feedback.get('confidence_score', None)
                batch_refs[orig_i].append(feedback)
                batch_ref_conf[orig_i].append(ref_conf)
                if r == self.MAX_REFLECTIONS -1:
                    continue
                trig = bool(feedback.get('trigger_reevaluation', False))
                print(f"    Q{orig_i}: trigger={trig}")
                if not trig:
                    continue

                # Re-evaluate with feedback injected
                reeval_start_time = time.time()
                user_query = batch_queries[orig_i]
                _, base_prompt = make_query(pd.Series(gold_rows[orig_i]), self.dataset)
                new_system_prompt = (
                    f"{base_prompt}\n\n## REFLECTION FEEDBACK ##\n"
                    f"Restart and improve using:\n"
                    f"Summary: {feedback.get('summary_comment', '')}\n"
                    f"Suggestions: {feedback.get('suggestions', '')}"
                )
                result2_msgs, tool_count2 = run_conversation_actor(
                    self.client, new_system_prompt, user_query, self.model,
                    self.tools, self.tool_mapping
                )
                reeval_time = time.time() - reeval_start_time

                try:
                    raw_content2 = result2_msgs[-1].get('content', '') if isinstance(result2_msgs[-1], dict) else ''
                    response2 = _best_effort_json(raw_content2) or {}
                except Exception as e:
                    print(f"Warning: failed to parse reevaluation output for item {orig_i}: {e}")
                    response2 = {}

                ok2 = evaluate_correctness(self.dataset, pd.Series(gold_rows[orig_i]), response2)
                conf2 = response2.get('confidence_score', None)

                batch_conf[orig_i].append(conf2)
                batch_results[orig_i].append(response2)
                batch_correct[orig_i].append(ok2)
                batch_tool_calls[orig_i].append(tool_count2)

                # Store reevaluation conversation history
                batch_conversation_histories[orig_i].append(result2_msgs)
                # Update timing for this item
                batch_times[orig_i] += reeval_time

                any_triggered = True


            if not any_triggered:
                break

        # Calculate batch time
        batch_end_time = time.time()
        batch_total_time = batch_end_time - batch_start_time

        # Get starting index for this batch
        start_idx = len(self.results)

        # Aggregate results
        self.results.extend(batch_results)
        self.correctness.extend(batch_correct)
        self.reflections.extend(batch_refs)
        self.confidences.extend(batch_conf)
        self.ref_confidences.extend(batch_ref_conf)
        self.tool_calls.extend(batch_tool_calls)
        self.processing_times.extend(batch_times)

        # Save histories incrementally to prevent memory buildup
        self._save_histories_incrementally(batch_conversation_histories, batch_reflection_histories, start_idx)

        # Keep only a small buffer in memory (last 100 items) to prevent crashes
        max_memory_items = 100
        self.conversation_histories.extend(batch_conversation_histories)
        self.reflection_histories.extend(batch_reflection_histories)

        if len(self.conversation_histories) > max_memory_items:
            # Keep only recent items in memory
            self.conversation_histories = self.conversation_histories[-max_memory_items:]
        if len(self.reflection_histories) > max_memory_items:
            self.reflection_histories = self.reflection_histories[-max_memory_items:]

        self._processed_count += len(batch_df)
        self._maybe_checkpoint()

        print(f"Batch {b_idx} completed in {batch_total_time:.2f}s (avg per item: {batch_total_time/len(batch_df):.2f}s)")

    def run(self):
        """Run the complete pipeline."""
        import time
        pipeline_start_time = time.time()
        print("Starting Actor Refinement Pipeline")

        # Load data
        self.load_data()

        # Build batches
        batches = self.prepare_batches()

        # Process all batches
        for b_idx, batch_indices in enumerate(batches, start=1):
            self.process_batch(batch_indices, b_idx)

        # Calculate total time
        total_time = time.time() - pipeline_start_time
        print(f"Total pipeline time: {total_time:.2f}s")

        # Final save
        self.save_checkpoint("final")

        print(f"[done] outputs in: {self.RUN_DIR}")
        return self.RUN_DIR


def main():
    """Main entry point with CLI argument parsing."""
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Actor Refinement Pipeline")
        parser.add_argument("--dataset", type=str, choices=["medqa","winogrande"], default="medqa")
        parser.add_argument("--split", type=str, choices=["train","dev","test"], default="test")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--embed_cache", type=str, default=None)
        parser.add_argument("--batching", type=str, choices=["kmeans","sequential","random"], default="sequential")
        parser.add_argument("--use_tools", action="store_true", help="Enable Brave tools")
        parser.add_argument("--out_dir", type=str, default="temp")
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--model",
            type=str,
            help=f"Model to use for inference. Supported models: {', '.join(SUPPORTED_MODELS[:5])}... (default: {DEFAULT_MODEL})"
        )
        args = parser.parse_args()

        if args.dataset not in args.out_dir:
            args.out_dir = f"outputs_{args.dataset}"

        pipeline = ActorRefinementPipeline(
            dataset=args.dataset,
            split=args.split,
            batch_size=args.batch_size,
            embed_cache=args.embed_cache,
            batching=args.batching,
            use_tools=args.use_tools,
            out_dir=args.out_dir,
            limit=args.limit,
            seed=args.seed,
            model=args.model
        )

        pipeline.run()

    except Exception as e:
        # Fallback to default values if argparse fails
        print(f"Using default configuration due to: {e}")
        pipeline = ActorRefinementPipeline(out_dir="outputs", model=DEFAULT_MODEL)
        pipeline.run()


if __name__ == "__main__":
    main()
