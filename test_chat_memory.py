import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import master_memory
from master_memory import (
    create_conversation,
    get_all_user_messages,
    initialize_database,
    list_conversations,
    load_conversation,
    search_past_chats,
    save_conversation,
)
from helper_functions import (
    get_chat_completion,
    get_chat_completion_with_usage,
    get_chat_completion_with_token_usage,
    summarize_user_chats,
)
from master_chat import choose_preferred_response, respond
from Archives.short_term_gradiochat import respond as respond_short_term


class ChatMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "memory.sqlite3"
        self.embedding_patcher = patch(
            "master_memory._embed_texts",
            side_effect=self._test_embeddings,
        )
        self.embedding_mock = self.embedding_patcher.start()
        initialize_database(self.database_path)

    def tearDown(self):
        self.embedding_patcher.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def _test_embeddings(texts):
        related_terms = {"art", "piece", "painting", "dog", "comet", "name"}
        return [
            [1.0, *([0.0] * 383)]
            if related_terms.intersection(re.findall(r"\w+", text.casefold()))
            else [0.0, 1.0, *([0.0] * 382)]
            for text in texts
        ]

    def test_conversation_persists_and_appears_in_history(self):
        conversation_id = create_conversation(self.database_path)
        messages = [
            {"role": "user", "content": "Explain gravity."},
            {"role": "assistant", "content": "Gravity attracts objects with mass."},
        ]

        save_conversation(conversation_id, messages, self.database_path)

        self.assertEqual(
            load_conversation(conversation_id, self.database_path), messages
        )
        self.assertEqual(
            list_conversations(self.database_path)[0][1], conversation_id
        )

    def test_all_saved_user_messages_are_returned_for_summary(self):
        first_chat = create_conversation(self.database_path)
        save_conversation(
            first_chat,
            [
                {"role": "user", "content": "I prefer concise explanations."},
                {"role": "assistant", "content": "Understood."},
            ],
            self.database_path,
        )
        second_chat = create_conversation(self.database_path)
        save_conversation(
            second_chat,
            [{"role": "user", "content": "I am learning Python."}],
            self.database_path,
        )

        messages = get_all_user_messages(self.database_path)

        self.assertEqual(
            [content for _, content in messages],
            ["I prefer concise explanations.", "I am learning Python."],
        )

    def test_search_finds_relevant_messages_across_saved_chats(self):
        first_chat = create_conversation(self.database_path)
        save_conversation(
            first_chat,
            [
                {"role": "user", "content": "My dog's name is Comet."},
                {"role": "assistant", "content": "I'll remember Comet."},
            ],
            self.database_path,
        )
        second_chat = create_conversation(self.database_path)
        save_conversation(
            second_chat,
            [{"role": "user", "content": "I enjoy astronomy."}],
            self.database_path,
        )

        results = search_past_chats(
            "What is my dog's name?", database_path=self.database_path
        )

        self.assertTrue(any("Comet" in content for _, _, content in results))
        self.assertFalse(any("astronomy" in content for _, _, content in results))

    def test_search_ranks_conceptually_similar_messages(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [
                {
                    "role": "user",
                    "content": "A painting I made for art class",
                },
            ],
            self.database_path,
        )
        unrelated_conversation = create_conversation(self.database_path)
        save_conversation(
            unrelated_conversation,
            [{"role": "user", "content": "A recipe for vegetable soup"}],
            self.database_path,
        )

        results = search_past_chats(
            "Tell me about my art piece",
            database_path=self.database_path,
        )
        self.assertEqual(len(results), 1)
        self.assertIn("painting", results[0][2])

    def test_keyword_search_finds_exact_terms_missed_by_vector_candidates(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "My instrument is a xylophone."}],
            self.database_path,
        )

        results = search_past_chats(
            "What instrument did I mention?",
            database_path=self.database_path,
        )

        self.assertTrue(any("xylophone" in content for _, _, content in results))

    def test_source_linked_claims_are_candidates_until_confirmed(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "I prefer concise explanations."}],
            self.database_path,
        )
        claim_id = master_memory.add_memory_claim(
            "Prefers concise explanations.",
            "preference",
            conversation_id,
            0,
            database_path=self.database_path,
        )

        self.assertFalse(
            any(
                result["kind"] == "claim"
                for result in master_memory.search_memory(
                    "concise explanations", database_path=self.database_path
                )
            )
        )
        master_memory.update_memory_claim(
            claim_id,
            "Prefers concise explanations.",
            "preference",
            "confirmed",
            database_path=self.database_path,
        )
        results = master_memory.search_memory(
            "concise explanations", database_path=self.database_path
        )

        self.assertTrue(any(result["kind"] == "claim" for result in results))
        claim = master_memory.list_memory_claims(self.database_path)[0]
        self.assertEqual(claim["source_conversation_id"], conversation_id)
        self.assertEqual(claim["source_message_index"], 0)

    def test_confirming_replacement_supersedes_prior_claim(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [
                {"role": "user", "content": "I prefer short answers."},
                {"role": "user", "content": "I now prefer detailed answers."},
            ],
            self.database_path,
        )
        original_id = master_memory.add_memory_claim(
            "Prefers short answers.",
            "preference",
            conversation_id,
            0,
            database_path=self.database_path,
        )
        master_memory.update_memory_claim(
            original_id,
            "Prefers short answers.",
            "preference",
            "confirmed",
            database_path=self.database_path,
        )
        replacement_id = master_memory.add_memory_claim(
            "Now prefers detailed answers.",
            "preference",
            conversation_id,
            1,
            supersedes_id=original_id,
            database_path=self.database_path,
        )
        conflict = master_memory.list_memory_conflicts(self.database_path)[0]
        self.assertEqual(conflict["status"], "open")

        self.assertEqual(
            master_memory.list_memory_claims(self.database_path)[0]["status"],
            "candidate",
        )
        master_memory.update_memory_claim(
            replacement_id,
            "Now prefers detailed answers.",
            "preference",
            "confirmed",
            database_path=self.database_path,
        )

        claims = {
            claim["id"]: claim
            for claim in master_memory.list_memory_claims(self.database_path)
        }
        self.assertEqual(claims[original_id]["status"], "superseded")
        self.assertEqual(claims[replacement_id]["status"], "confirmed")
        self.assertEqual(master_memory.list_memory_conflicts(self.database_path), [])
        with sqlite3.connect(self.database_path) as connection:
            conflict_status = connection.execute(
                "SELECT status FROM memory_conflicts WHERE id = ?",
                (conflict["id"],),
            ).fetchone()[0]
        self.assertEqual(conflict_status, "resolved")

    def test_confirmed_claims_are_searchable_by_semantic_similarity(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "My dog's name is Comet."}],
            self.database_path,
        )
        claim_id = master_memory.add_memory_claim(
            "The user's dog is named Comet.",
            "pet identity",
            conversation_id,
            0,
            database_path=self.database_path,
        )
        master_memory.update_memory_claim(
            claim_id,
            "The user's dog is named Comet.",
            "pet identity",
            "confirmed",
            database_path=self.database_path,
        )

        results = master_memory.search_memory(
            "What is my dog's name?",
            database_path=self.database_path,
            retrieval_mode="hybrid",
        )

        self.assertTrue(
            any(result["kind"] == "claim" and result["id"] == claim_id for result in results)
        )

    def test_related_claims_are_retrieved_with_a_bounded_graph_walk(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [
                {"role": "user", "content": "My dog is named Comet."},
                {"role": "user", "content": "Comet's breed is collie."},
                {"role": "user", "content": "Comet likes hiking."},
            ],
            self.database_path,
        )
        claim_ids = []
        for index, content in enumerate(
            ("My dog is named Comet.", "Comet's breed is collie.", "Comet likes hiking.")
        ):
            claim_id = master_memory.add_memory_claim(
                content,
                "pet profile",
                conversation_id,
                index,
                database_path=self.database_path,
            )
            master_memory.update_memory_claim(
                claim_id, content, "pet profile", "confirmed", self.database_path
            )
            claim_ids.append(claim_id)
        master_memory.link_memory_claims(
            claim_ids[0], claim_ids[1], "has breed", self.database_path
        )
        master_memory.link_memory_claims(
            claim_ids[1], claim_ids[2], "likes activity", self.database_path
        )

        with patch(
            "master_memory._rank_confirmed_claims",
            return_value=[-claim_ids[0]],
        ), patch("master_memory._vector_rank", return_value=[]):
            results = master_memory.search_memory(
                "unmatched term",
                database_path=self.database_path,
                retrieval_mode="hybrid",
                limit=10,
            )

        related = [result for result in results if result["kind"] == "related_claim"]
        self.assertEqual(
            {result["id"] for result in related},
            {claim_ids[1], claim_ids[2]},
        )
        self.assertEqual(max(result["relation_depth"] for result in related), 2)

    def test_potential_conflicts_are_flagged_without_overwriting_claims(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [
                {"role": "user", "content": "I prefer concise explanations."},
                {"role": "user", "content": "I prefer detailed explanations."},
            ],
            self.database_path,
        )
        for index, content in enumerate(
            ("Prefers concise explanations.", "Prefers detailed explanations.")
        ):
            master_memory.add_memory_claim(
                content,
                "communication preference",
                conversation_id,
                index,
                database_path=self.database_path,
            )

        conflicts = master_memory.list_memory_conflicts(self.database_path)

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["status"], "open")
        claims = master_memory.list_memory_claims(self.database_path)
        self.assertEqual({claim["status"] for claim in claims}, {"candidate"})

    def test_fresh_conversation_summaries_guide_retrieval(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "I enjoy astronomy."}],
            self.database_path,
        )
        master_memory.save_conversation_summary(
            conversation_id,
            "The user likes taking telescope photographs of Saturn.",
            self.database_path,
        )

        with patch("master_memory._vector_rank", return_value=[]):
            results = master_memory.search_memory(
                "telescope photographs",
                database_path=self.database_path,
            )

        self.assertTrue(any("astronomy" in result["content"] for result in results))
        save_conversation(
            conversation_id,
            [
                {"role": "user", "content": "I enjoy astronomy."},
                {"role": "assistant", "content": "Tell me about Saturn."},
            ],
            self.database_path,
        )
        self.assertTrue(
            master_memory.list_conversation_summaries(self.database_path)[0]["stale"]
        )
        with patch("master_memory._vector_rank", return_value=[]):
            results = master_memory.search_memory(
                "telescope photographs",
                database_path=self.database_path,
            )
        self.assertEqual(results, [])

    def test_retrieval_metrics_store_performance_and_feedback_not_query_text(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "My private project is called Cedar."}],
            self.database_path,
        )

        results, event_id = master_memory.search_memory(
            "What is my private project called?",
            database_path=self.database_path,
            include_event_id=True,
        )
        self.assertTrue(results)
        result_key = results[0]["feedback_key"]
        master_memory.record_retrieval_feedback(
            event_id, result_key, True, database_path=self.database_path
        )
        for _ in range(2):
            _, repeat_event_id = master_memory.search_memory(
                "What is my private project called?",
                database_path=self.database_path,
                include_event_id=True,
            )
            master_memory.record_retrieval_feedback(
                repeat_event_id, result_key, True, database_path=self.database_path
            )
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT elapsed_ms, index_size, strategy "
                "FROM retrieval_metrics WHERE id = ?",
                (event_id,),
            ).fetchone()
            ratings = connection.execute(
                "SELECT COUNT(*) FROM retrieval_metric_results "
                "WHERE record_key = ? AND feedback = 1",
                (result_key,),
            ).fetchone()[0]
            columns = {
                column[1]
                for column in connection.execute(
                    "PRAGMA table_info(retrieval_metrics)"
                )
            }
        self.assertGreaterEqual(row[0], 0)
        self.assertEqual(row[1:], (1, "exact"))
        self.assertEqual(ratings, 3)
        self.assertNotIn("query", columns)
        self.assertGreater(
            master_memory._feedback_adjustment(
                master_memory.hashlib.sha256(
                    b"what is my private project called?"
                ).hexdigest(),
                result_key,
                self.database_path,
            ),
            1.0,
        )

    def test_repeated_per_item_feedback_changes_same_query_ranking(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [
                {"role": "user", "content": "Comet is my dog."},
                {"role": "user", "content": "My dog is Comet."},
            ],
            self.database_path,
        )
        query = "Comet dog"
        initial = master_memory.search_memory(
            query, database_path=self.database_path, limit=2
        )
        self.assertEqual(len(initial), 2)
        preferred_key = initial[1]["feedback_key"]
        for _ in range(3):
            _, event_id = master_memory.search_memory(
                query,
                database_path=self.database_path,
                limit=2,
                include_event_id=True,
                use_feedback=False,
            )
            master_memory.record_retrieval_feedback(
                event_id,
                preferred_key,
                True,
                database_path=self.database_path,
            )

        updated = master_memory.search_memory(
            query, database_path=self.database_path, limit=2
        )

        self.assertEqual(updated[0]["feedback_key"], preferred_key)

    def test_labeled_benchmark_reports_recall_mrr_and_latency(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [
                {"role": "user", "content": "My dog's name is Comet."},
                {"role": "assistant", "content": "Comet likes hiking."},
            ],
            self.database_path,
        )

        report = master_memory.evaluate_retrieval(
            [
                {
                    "query": "What is my dog's name?",
                    "relevant_sources": [(conversation_id, 0)],
                }
            ],
            database_path=self.database_path,
        )

        for mode in ("semantic", "hybrid"):
            self.assertEqual(report[mode]["case_count"], 1)
            self.assertGreaterEqual(report[mode]["recall_at_5"], 0)
            self.assertGreaterEqual(report[mode]["mean_reciprocal_rank"], 0)
            self.assertGreaterEqual(report[mode]["mean_latency_ms"], 0)
            self.assertEqual(len(report[mode]["latencies_ms"]), 1)


    def test_deleting_conversation_removes_source_memories_and_search_records(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "My favorite color is teal."}],
            self.database_path,
        )
        claim_id = master_memory.add_memory_claim(
            "Favorite color is teal.",
            "preference",
            conversation_id,
            0,
            database_path=self.database_path,
        )
        master_memory.delete_conversation(conversation_id, self.database_path)

        self.assertEqual(master_memory.list_memory_claims(self.database_path), [])
        self.assertEqual(
            search_past_chats("favorite color", database_path=self.database_path),
            [],
        )
        with sqlite3.connect(self.database_path) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM memory_claims WHERE id = ?", (claim_id,)
                ).fetchone()
            )

    def test_long_messages_are_split_and_retrieved_by_relevant_chunk(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [
                {
                    "role": "user",
                    "content": "unrelated details " * 300
                    + "I painted a landscape for my art class.",
                },
            ],
            self.database_path,
        )

        results = search_past_chats(
            "Tell me about my art piece",
            database_path=self.database_path,
        )

        self.assertEqual(len(results), 1)
        self.assertIn("painted a landscape", results[0][2])

    def test_saving_appended_message_reuses_existing_embeddings(self):
        conversation_id = create_conversation(self.database_path)
        initial_messages = [{"role": "user", "content": "I enjoy astronomy."}]
        save_conversation(conversation_id, initial_messages, self.database_path)
        self.embedding_mock.reset_mock()

        save_conversation(
            conversation_id,
            initial_messages
            + [{"role": "assistant", "content": "Stars are fascinating."}],
            self.database_path,
        )

        self.embedding_mock.assert_called_once_with(["Stars are fascinating."])

    def test_ann_search_uses_index_candidates_not_all_embeddings(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "My painting is in the studio."}],
            self.database_path,
        )
        statements = []
        connect = master_memory._connect
        with patch(
            "master_memory._connect",
            side_effect=lambda path: self._trace_connection(
                connect(path), statements
            ),
        ):
            results = search_past_chats(
                "Tell me about my art piece",
                database_path=self.database_path,
            )

        retrieval_queries = [
            statement
            for statement in statements
            if "FROM chat_message_embeddings" in statement
            and "JOIN conversations" in statement
        ]
        self.assertEqual(len(retrieval_queries), 1)
        self.assertIn("WHERE chat_message_embeddings.id IN", retrieval_queries[0])
        self.assertEqual(len(results), 1)

    def test_ann_index_replays_saved_changes_after_restart(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "My painting is in the studio."}],
            self.database_path,
        )
        cache_key = str(self.database_path.resolve())
        master_memory._ann_indexes.pop(cache_key)

        results = search_past_chats(
            "Tell me about my art piece",
            database_path=self.database_path,
        )

        self.assertEqual(len(results), 1)
        self.assertIn("painting", results[0][2])

    def test_ann_index_rebuilds_and_checkpoints_after_many_changes(self):
        with patch("master_memory._ANN_CHECKPOINT_INTERVAL", 2):
            for content in (
                "My painting is in the studio.",
                "I enjoy astronomy.",
            ):
                conversation_id = create_conversation(self.database_path)
                save_conversation(
                    conversation_id,
                    [{"role": "user", "content": content}],
                    self.database_path,
                )

        with sqlite3.connect(self.database_path) as connection:
            revision = connection.execute(
                "SELECT revision FROM chat_embedding_state WHERE id = 1"
            ).fetchone()[0]
            checkpoint_revision = connection.execute(
                "SELECT revision FROM chat_ann_checkpoint WHERE id = 1"
            ).fetchone()[0]
            change_count = connection.execute(
                "SELECT COUNT(*) FROM chat_ann_changes"
            ).fetchone()[0]

        self.assertEqual(checkpoint_revision, revision)
        self.assertEqual(change_count, 0)

    def test_ann_index_removes_vectors_for_changed_messages(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "My painting is in the studio."}],
            self.database_path,
        )
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "I enjoy astronomy."}],
            self.database_path,
        )

        results = search_past_chats(
            "Tell me about my art piece",
            database_path=self.database_path,
        )

        self.assertEqual(results, [])

    def test_embedding_storage_upgrade_is_compact_without_reembedding(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "I enjoy astronomy."}],
            self.database_path,
        )
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE chat_message_embeddings SET embedding = ?",
                (json.dumps([1.0, *([0.0] * 383)]),),
            )
            connection.execute(
                "UPDATE chat_embedding_state SET index_version = 3 WHERE id = 1"
            )
            connection.execute("DELETE FROM chat_ann_checkpoint")
        master_memory._ann_indexes.pop(str(self.database_path.resolve()), None)
        self.embedding_mock.reset_mock()

        initialize_database(self.database_path)

        self.embedding_mock.assert_not_called()
        with sqlite3.connect(self.database_path) as connection:
            storage_type, byte_count = connection.execute(
                "SELECT typeof(embedding), length(embedding) "
                "FROM chat_message_embeddings"
            ).fetchone()
        self.assertEqual((storage_type, byte_count), ("blob", 384 * 4))

    @staticmethod
    def _trace_connection(connection, statements):
        connection.set_trace_callback(statements.append)
        return connection

    def test_initialization_indexes_existing_conversations(self):
        conversation_id = create_conversation(self.database_path)
        save_conversation(
            conversation_id,
            [{"role": "user", "content": "My favorite color is teal."}],
            self.database_path,
        )
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DROP TABLE chat_message_embeddings")
            connection.execute(
                """
                CREATE TABLE chat_message_embeddings (
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )
            connection.execute("DROP TABLE chat_embedding_state")
            connection.execute(
                """
                CREATE TABLE chat_embedding_state (
                    id INTEGER PRIMARY KEY,
                    model TEXT NOT NULL,
                    indexed INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO chat_embedding_state (id, model, indexed)
                VALUES (1, 'sentence-transformers/all-MiniLM-L6-v2', 1)
                """
            )

        initialize_database(self.database_path)

        results = search_past_chats("favorite color", database_path=self.database_path)
        self.assertTrue(any("teal" in content for _, _, content in results))

    def test_summary_processes_all_messages_in_bounded_chunks(self):
        calls = []

        def create_completion(**kwargs):
            calls.append(
                (
                    kwargs["messages"][0]["content"],
                    kwargs["messages"][1]["content"],
                )
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=f"summary {len(calls)}")
                    )
                ]
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create_completion)
            )
        )
        conversations = [
            ("First", "alpha " * 50),
            ("Second", "beta " * 50),
            ("Third", "gamma " * 50),
        ]

        with patch("helper_functions._get_client", return_value=client):
            summary = summarize_user_chats(conversations, max_chunk_chars=128)

        self.assertTrue(summary.startswith("summary"))
        self.assertGreater(len(calls), 3)
        first_stage_calls = [
            content for instruction, content in calls if instruction.startswith("Extract")
        ]
        combined_input = "\n".join(first_stage_calls)
        self.assertIn("alpha", combined_input)
        self.assertIn("beta", combined_input)
        self.assertIn("gamma", combined_input)

    def test_both_models_share_context_and_preferred_answer_is_saved(self):
        history = [{"role": "user", "content": "Earlier question"}]
        pending = {
            "prompt": "What should I do next?",
            "history": history,
            "conversation_id": "existing-chat",
        }
        with (
            patch(
                "master_chat.search_memory",
                return_value=(
                    [
                        {
                            "kind": "message",
                            "title": "Earlier chat",
                            "date": "2026-01-01",
                            "source_message_index": 0,
                            "role": "user",
                            "content": "I am learning Python.",
                            "source_excerpt": "I am learning Python.",
                            "feedback_key": "message:101",
                        }
                    ],
                    42,
                    [("Past chat: I am learning Python.", "message:101")],
                ),
            ),
            patch(
                "master_chat.get_chat_completion_with_token_usage",
                side_effect=[
                    ("Luna draft", 100, 20),
                    ("5-mini draft", 110, 30),
                ],
            ) as complete,
            patch(
                "master_chat.list_conversations",
                return_value=[("Existing chat", "existing-chat")],
            ),
        ):
            response = respond(
                pending["prompt"], history, "existing-chat", "Both", 75
            )

        self.assertEqual(complete.call_count, 2)
        self.assertEqual(
            [call.args[3] for call in complete.call_args_list],
            ["do-6-luna", "do-5-mini"],
        )
        self.assertEqual(
            complete.call_args_list[0].args[2],
            complete.call_args_list[1].args[2],
        )
        self.assertEqual(response[5:7], ("Luna draft", "5-mini draft"))
        self.assertIn("Tokens from latest input:** 210", response[11])
        self.assertIn("Tokens from latest output:** 50", response[11])
        self.assertIn("Total tokens used this session:** 335", response[11])
        self.assertEqual(response[12], 335)
        pending["conversation_id"] = response[2]
        pending["token_usage_text"] = response[11]
        pending["session_token_total"] = response[12]

        with (
            patch("master_chat.save_conversation") as save,
            patch(
                "master_chat.list_conversations",
                return_value=[("Existing chat", "existing-chat")],
            ),
        ):
            selected = choose_preferred_response(
                "5-mini", "Luna draft", "5-mini draft", pending
            )

        self.assertEqual(
            save.call_args.args,
            (
                "existing-chat",
                [
                    {"role": "user", "content": "Earlier question"},
                    {"role": "user", "content": "What should I do next?"},
                    {"role": "assistant", "content": "5-mini draft"},
                ],
            ),
        )
        self.assertEqual(selected[0][-1]["content"], "5-mini draft")
        self.assertEqual(selected[11], response[11])
        self.assertEqual(selected[12], response[12])

    def test_single_model_chat_displays_api_token_usage(self):
        with (
            patch(
                "master_chat._past_chat_context",
                return_value=("retrieved context", "shown evidence", 7, []),
            ),
            patch(
                "master_chat.get_chat_completion_with_token_usage",
                return_value=("Model response", 31, 8),
            ),
            patch("master_chat.create_conversation", return_value="new-chat"),
            patch("master_chat.save_conversation"),
            patch(
                "master_chat.list_conversations",
                return_value=[("New chat", "new-chat")],
            ),
        ):
            response = respond("Hello", [], None, "5-mini", 50)

        self.assertEqual(response[0][-1]["content"], "Model response")
        self.assertIn("Tokens from latest input:** 31", response[11])
        self.assertIn("Tokens from latest output:** 8", response[11])
        self.assertIn("Total tokens used this session:** 89", response[11])
        self.assertEqual(response[12], 89)

    def test_chat_completion_uses_requested_model(self):
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Model response")
                )
            ],
            usage=SimpleNamespace(prompt_tokens=23),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=completion))
            )
        )

        with patch("helper_functions._get_client", return_value=client):
            answer = get_chat_completion("Hello", [], model="do-6-luna")

        self.assertEqual(answer, "Model response")
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"],
            "do-6-luna",
        )

    def test_chat_completion_returns_input_token_usage(self):
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Model response")
                )
            ],
            usage=SimpleNamespace(prompt_tokens=23),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=completion))
            )
        )

        with patch("helper_functions._get_client", return_value=client):
            answer, input_tokens = get_chat_completion_with_usage("Hello", [])

        self.assertEqual(answer, "Model response")
        self.assertEqual(input_tokens, 23)

    def test_chat_completion_returns_input_and_output_token_usage(self):
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Model response"))
            ],
            usage=SimpleNamespace(prompt_tokens=23, completion_tokens=7),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=completion))
            )
        )

        with patch("helper_functions._get_client", return_value=client):
            answer, input_tokens, output_tokens = get_chat_completion_with_token_usage(
                "Hello", []
            )

        self.assertEqual(answer, "Model response")
        self.assertEqual((input_tokens, output_tokens), (23, 7))

    def test_token_usage_includes_citation_retry(self):
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="Comet"))
                ],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=4),
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Comet is your dog [M1].")
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=35, completion_tokens=9),
            ),
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(side_effect=responses)
                )
            )
        )

        with patch("helper_functions._get_client", return_value=client):
            answer, input_tokens, output_tokens = get_chat_completion_with_token_usage(
                "What is my dog's name?",
                [],
                "[M1] User said their dog's name is Comet.",
            )

        self.assertIn("[M1]", answer)
        self.assertEqual((input_tokens, output_tokens), (55, 13))

    def test_chat_completion_retries_until_retrieved_memory_is_cited(self):
        responses = [
            "Your dog's name is Comet.",
            "Your dog's name is Comet [M1].",
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        side_effect=[
                            SimpleNamespace(
                                choices=[
                                    SimpleNamespace(
                                        message=SimpleNamespace(content=response)
                                    )
                                ]
                            )
                            for response in responses
                        ]
                    )
                )
            )
        )

        with patch("helper_functions._get_client", return_value=client):
            answer = get_chat_completion(
                "What is my dog's name?",
                [],
                "[M1] User said their dog's name is Comet.",
            )

        self.assertIn("[M1]", answer)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_chat_completion_rejects_retrieval_answers_without_valid_citation(self):
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="No citation"))
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=completion))
            )
        )

        with patch("helper_functions._get_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "valid citation"):
                get_chat_completion(
                    "What is my dog's name?",
                    [],
                    "[M1] User said their dog's name is Comet.",
                )

    def test_short_term_chat_displays_input_tokens_without_adding_them_to_history(self):
        with patch(
            "Archives.short_term_gradiochat.get_chat_completion_with_usage",
            return_value=("Model response", 23),
        ):
            history, cleared_message, token_count = respond_short_term("Hello", [])

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Model response"},
            ],
        )
        self.assertEqual(cleared_message, "")
        self.assertIn("**23**", token_count)


if __name__ == "__main__":
    unittest.main()
