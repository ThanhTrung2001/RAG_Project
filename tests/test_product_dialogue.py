"""Offline regressions: no embedding downloads, catalog writes, or Ollama calls.

Run: python -m unittest discover -s tests -v
"""
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from product_dialogue import route_product_query, resolve_product_references


SOURCES = [{"id": i, "caption": f"Tent model {i}", "score": 0.1,
            "image_path": f"data/images/{i:06d}.jpg"} for i in range(5)]


class RoutingTests(unittest.TestCase):
    def test_intent_and_identity(self):
        cases = [
            ("Sản phẩm số 2 chất liệu gì?", None, "detail", [1]),
            ("What is the weight of [5]?", None, "detail", [4]),
            ("Show me details of product 2", None, "detail", [1]),
            ("Compare product 1 and product 2", None, "detail", [0, 1]),
            ("Is product 2 cheaper than product 1?", None, "detail", [1, 0]),
            ("Cái này chất liệu gì?", 0, "detail", [0]),
            ("Chất liệu gì?", 0, "detail", [0]),
            ("How much capacity?", 1, "detail", [1]),
            ("Is it waterproof?", None, "clarify", []),
            ("Cái đó chất liệu gì?", None, "clarify", []),
            ("Product 99 details?", None, "clarify", []),
            ("Tìm lều chống nước", 0, "search", []),
            ("Recommend a coffee maker", 0, "search", []),
            ("Something cheaper?", 0, "search", []),
            ("Show me more options like that", 0, "search", []),
            ("Có màu khác không?", 0, "search", []),
        ]
        for query, focus, intent, ids in cases:
            with self.subTest(query=query):
                route = route_product_query(query, SOURCES, focus)
                self.assertEqual((route["intent"], route["product_ids"]), (intent, ids))

    def test_no_context_does_not_guess(self):
        self.assertEqual(route_product_query("Is it waterproof?", [])["intent"], "clarify")
        self.assertEqual(route_product_query("first aid kit", [])["intent"], "search")

    def test_new_topic_with_focus_uses_classifier(self):
        classify = Mock(return_value='{"intent":"search","product_ids":[]}')
        self.assertEqual(route_product_query("a waterproof jacket", SOURCES, 0, classify)["intent"], "search")
        classify.assert_called_once()

    def test_classifier_cannot_invent_ids(self):
        for raw in ['{"intent":"detail","product_ids":[99]}',
                    '{"intent":"detail","product_ids":[true]}',
                    '{"intent":"detail","product_ids":[]}', 'not json', '[]']:
            with self.subTest(raw=raw):
                self.assertEqual(route_product_query("Tell me about those", SOURCES,
                                 llm_call=lambda _: raw)["intent"], "clarify")

    def test_classifier_resolves_known_product(self):
        result = route_product_query("Would this suit my trip?", SOURCES, 0,
                                     lambda _: '```json\n{"intent":"detail","product_ids":[0]}\n```')
        self.assertEqual(result["product_ids"], [0])

    def test_short_followups_use_focus_before_classifier(self):
        for query in ["How much capacity?", "How does it monitor air quality?"]:
            with self.subTest(query=query):
                classify = Mock(return_value='{"intent":"detail","product_ids":[0,1]}')
                result = route_product_query(query, SOURCES[:2], 1, classify)
                self.assertEqual(result["product_ids"], [1])
                self.assertEqual(result["reason"], "rules")
                classify.assert_not_called()
        self.assertEqual(route_product_query("How does it monitor air quality?", SOURCES[:2])["intent"], "clarify")
        self.assertEqual(route_product_query("How does it monitor air quality?", SOURCES[:1])["product_ids"], [0])

    def test_resolve_positions_before_local_citation_numbering(self):
        for query in ["Show me details of product 2", "Tell me about the second product",
                      "Chi tiết sản phẩm số 2", "Chi tiết cái 2", "Details of [2]"]:
            with self.subTest(query=query):
                resolved = resolve_product_references(query, SOURCES[:2])
                self.assertIn(SOURCES[1]["caption"], resolved)
                self.assertNotIn("product 2", resolved)
        self.assertEqual(resolve_product_references("Compare product 2 and product 1", SOURCES),
                         "Compare Tent model 1 and Tent model 0")


class DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import real RAG implementation, replacing only its heavyweight search dependency.
        fake = types.ModuleType("search_core")
        fake.search = Mock()
        fake.rrf_fusion = Mock()
        fake.get_product = Mock()
        spec = importlib.util.spec_from_file_location("rag_core_offline_test",
                 Path(__file__).resolve().parents[1] / "rag_core.py")
        cls.rag = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"search_core": fake}):
            spec.loader.exec_module(cls.rag)

    def setUp(self):
        self.rag._SESSION_STORE.clear()
        self.rag._PRODUCT_STATE.clear()
        self.products = {s["id"]: {**s, "brand": "Test Brand", "category": "Camping",
                                  "description": "Waterproof fabric; weight 2 kg."} for s in SOURCES}
        self.catalog = patch.object(self.rag, "get_product", side_effect=self.products.get).start()
        self.llm = patch.object(self.rag, "call_llm", return_value="Weight: 2 kg [1].").start()
        self.retrieve = patch.object(self.rag, "retrieve", return_value=SOURCES).start()
        self.addCleanup(patch.stopall)

    def seed(self, session="s"):
        self.rag.append_history(session, "recommend tents", "Here are tents", SOURCES)

    def test_detail_bypasses_search_in_both_pipelines(self):
        for fn in [self.rag.answer, self.rag.answer_agentic]:
            with self.subTest(fn=fn.__name__):
                self.seed()
                result = fn("Product 5 weight?", session_id="s")
                self.assertEqual(result["intent"], "detail")
                self.assertEqual(result["sources"][0]["id"], 4)
                self.assertIn("weight 2 kg", self.llm.call_args.args[0])
                self.assertIn("Test Brand", self.llm.call_args.args[0])
                self.assertIn("Customer question: Tent model 4 weight?", self.llm.call_args.args[0])
                self.retrieve.assert_not_called()

    def test_focus_survives_multiple_detail_turns(self):
        for fn in [self.rag.answer, self.rag.answer_agentic]:
            with self.subTest(fn=fn.__name__):
                self.rag.append_history("s", "recommend tents", "Here are tents", SOURCES[:2])
                selected_id = None
                for position in [1, 2, 1]:
                    result = fn(f"Chi tiết cái {position}", session_id="s",
                                selected_product_id=selected_id)
                    self.assertEqual(result["intent"], "detail")
                    self.assertEqual(result["sources"][0]["id"], position - 1)
                    self.assertEqual(result["product_options"], SOURCES[:2])
                    for _ in range(4):
                        result = fn("Chất liệu gì?", session_id="s")
                        self.assertEqual(result["sources"][0]["id"], position - 1)
                        self.assertEqual(result["product_options"], SOURCES[:2])
                    selected_id = position - 1
                self.assertEqual(len(self.rag.get_history("s")), 3)
        self.retrieve.assert_not_called()

    def test_happy_conversation_scenarios(self):
        scenarios = json.loads(Path(__file__).with_name("product_dialogue_scenarios.json")
                               .read_text(encoding="utf-8"))["scenarios"]
        for scenario in scenarios:
            for mode in ["naive", "agentic"]:
                for simulate_ui in [False, True]:
                    with self.subTest(scenario=scenario["id"], mode=mode, ui=simulate_ui):
                        self._run_happy_scenario(scenario, mode, simulate_ui)

    def _run_happy_scenario(self, scenario, mode, simulate_ui):
        self.rag.clear_history("scenario")
        result_sets = {"initial": SOURCES[:2], "alternatives": SOURCES[2:4],
                       "topic": SOURCES[3:5], "single": SOURCES[:1],
                       "one_recommended": SOURCES[:2], "first_recommended": SOURCES[:2],
                       "topic_first": SOURCES[3:5]}
        fn = self.rag.answer if mode == "naive" else self.rag.answer_agentic
        options, focus, selected = [], None, None
        for turn, step in enumerate(scenario["steps"], 1):
            expected_ids = [options[n - 1]["id"] for n in step.get("targets", [])]
            self.retrieve.reset_mock()
            self.retrieve.return_value = result_sets.get(step.get("result_set"), [])
            self.llm.reset_mock()

            def llm_response(prompt):
                if prompt.startswith("Classify the customer's intent"):
                    # The classifier is a controlled fixture, not an accuracy claim.
                    return json.dumps({"intent": step["intent"], "product_ids": expected_ids})
                if step["intent"] == "search":
                    if step.get("result_set") == "one_recommended":
                        return "I recommend only the second product [2]."
                    if step.get("result_set") in {"first_recommended", "topic_first"}:
                        return "I recommend only the first product [1]."
                    return "Suggested products: " + " ".join(
                        f"[{i}]" for i in range(1, len(self.retrieve.return_value) + 1))
                if prompt.startswith("Answer the customer's question about ONLY"):
                    for pid in expected_ids:
                        self.assertIn(self.products[pid]["caption"], prompt)
                    for pid in set(self.products) - set(expected_ids):
                        self.assertNotIn(self.products[pid]["caption"], prompt)
                return "Catalog answer [1]."

            self.llm.side_effect = llm_response
            with patch.object(self.rag, "contextualize_query", return_value="scenario search"), \
                 patch.object(self.rag, "judge_sufficiency", return_value="YES") as judge:
                response = fn(step["query"], session_id="scenario",
                              selected_product_id=selected if simulate_ui else None,
                              image=object() if step.get("image") else None)
                label = f'{scenario["id"]}/{mode}/ui={simulate_ui}/turn={turn}: {step["query"]}'
                self.assertEqual(response["intent"], step["intent"], label)
                if step["intent"] == "search":
                    options = result_sets[step["result_set"]]
                    focus = (options[0]["id"] if len(options) == 1 else
                             options[1]["id"] if step.get("result_set") == "one_recommended" else
                             options[0]["id"] if step.get("result_set") in {"first_recommended", "topic_first"} else None)
                    selected = focus
                    self.assertEqual(response["focused_product_id"], focus, label)
                    self.assertEqual(response["sources"], options, label)
                    self.retrieve.assert_called_once()
                    self.assertEqual(self.retrieve.call_args.kwargs["image"] is not None,
                                     bool(step.get("image")), label)
                    self.assertEqual(judge.call_count, int(mode == "agentic"), label)
                elif step["intent"] == "detail":
                    self.assertEqual([s["id"] for s in response["sources"]], expected_ids, label)
                    self.assertEqual(response["product_options"], options, label)
                    focus = expected_ids[0] if len(expected_ids) == 1 else None
                    selected = focus
                    self.retrieve.assert_not_called()
                    judge.assert_not_called()
                else:
                    self.assertEqual(response["product_options"], options, label)
                    self.retrieve.assert_not_called()
                    judge.assert_not_called()
                state = self.rag._PRODUCT_STATE["scenario"]
                self.assertEqual(state["sources"], options, label)
                self.assertEqual(state["focused_id"], focus, label)
                self.assertLessEqual(len(self.rag.get_history("scenario")), 3, label)
                if step.get("image"):
                    self.assertTrue(self.rag.get_history("scenario")[-1]["query"].startswith("[image]"), label)
        self.llm.side_effect = None

    def test_selection_zero_and_older_card(self):
        self.seed()
        self.rag.answer("Product 5 weight?", session_id="s")
        result = self.rag.answer("Cái này chất liệu gì?", session_id="s", selected_product_id=0)
        self.assertEqual(result["sources"][0]["id"], 0)
        self.assertEqual(result["product_options"], SOURCES)

    def test_clarify_preserves_options_and_does_not_call_llm(self):
        self.seed()
        result = self.rag.answer("Cái đó chất liệu gì?", session_id="s")
        self.assertEqual(result["intent"], "clarify")
        self.assertEqual(len(result["sources"]), 5)
        self.llm.assert_not_called()
        result = self.rag.answer("Product 4 details?", session_id="s")
        self.assertEqual(result["sources"][0]["id"], 3)

    def test_new_search_clears_focus(self):
        for fn in [self.rag.answer, self.rag.answer_agentic]:
            for sources, answer, expected in [
                (SOURCES, "Suggestions [1] and [2]", None),
                (SOURCES, "Recommend [2], more about it [2]", 1),
                (SOURCES, "Only [1]", 0),
                (SOURCES, "No citation", None),
                (SOURCES, "Invalid [99] [0]", None),
                (SOURCES[:1], "One product without citation", 0),
                ([], "No products", None),
            ]:
                with self.subTest(fn=fn.__name__, answer=answer):
                    self.seed()
                    self.retrieve.reset_mock()
                    self.retrieve.return_value = sources
                    self.llm.return_value = answer
                    with patch.object(self.rag, "contextualize_query", return_value="coffee maker"), \
                         patch.object(self.rag, "judge_sufficiency", return_value="YES"):
                        result = fn("Recommend a coffee maker", session_id="s", selected_product_id=0)
                    self.assertEqual(result["intent"], "search")
                    self.retrieve.assert_called_once()
                    self.assertEqual(result["focused_product_id"], expected)
                    self.assertEqual(self.rag._PRODUCT_STATE["s"]["focused_id"], expected)
                    self.assertEqual(self.rag._PRODUCT_STATE["s"]["sources"], sources)

    def test_reset_and_session_isolation(self):
        self.seed()
        result = self.rag.answer("Is it waterproof?", session_id="other")
        self.assertEqual(result["intent"], "clarify")
        self.rag.clear_history("s")
        self.assertNotIn("s", self.rag._PRODUCT_STATE)
        self.assertEqual(self.rag.get_history("s"), [])

    def test_baseline_switch_skips_router(self):
        self.seed()
        with patch.object(self.rag, "contextualize_query", return_value="tent"), \
             patch.object(self.rag, "_answer_product_followup") as router:
            self.rag.answer("Product 2 weight?", session_id="s", product_dialogue=False)
        router.assert_not_called()
        self.retrieve.assert_called_once()

    def test_missing_attributes_and_description_truncation(self):
        self.seed()
        self.products[0]["description"] = ""
        self.products[0]["description_truncated"] = True
        self.rag.answer("Product 1 price?", session_id="s")
        prompt = self.llm.call_args.args[0]
        self.assertIn("catalog\ndoes not provide it", prompt)
        self.assertIn("description truncated", prompt)
        self.assertNotIn("weight 2 kg", prompt)

    def test_image_stays_on_search_path(self):
        with patch.object(self.rag, "_answer_product_followup") as router:
            result = self.rag.answer("", image=object(), session_id="s")
        router.assert_not_called()
        self.assertEqual(result["intent"], "search")

    def test_image_text_retrieval_fuses_branches_without_reranking(self):
        image = object()
        with patch.object(self.rag, "search", side_effect=[SOURCES[:2], SOURCES[1:3]]) as search, \
             patch.object(self.rag, "rrf_fusion", return_value=[(1, 0.0328), (2, 0.016)]) as fusion:
            results = self.rag._retrieve_with_image("air fryer", image, 2)
        self.assertEqual([r["id"] for r in results], [1, 2])
        self.assertEqual(search.call_count, 2)
        self.assertIs(search.call_args_list[0].args[0], image)
        self.assertEqual(search.call_args_list[0].kwargs["query_type"], "image")
        self.assertEqual(search.call_args_list[1].args[0], "air fryer")
        self.assertEqual(search.call_args_list[1].kwargs["query_type"], "text")
        for call in search.call_args_list:
            self.assertEqual(call.kwargs["k"], 50)
            self.assertNotIn("reranker", call.kwargs)
        fusion.assert_called_once_with([(0, 0.1), (1, 0.1)], [(1, 0.1), (2, 0.1)])

    def test_agentic_image_rewrite_keeps_image_and_skips_reranker(self):
        image = object()
        with patch.object(self.rag, "judge_sufficiency", side_effect=["NO: need another result", "YES"]), \
             patch.object(self.rag, "rewrite_query", return_value="rewritten air fryer"), \
             patch.object(self.rag, "_answer_product_followup") as router:
            result = self.rag.answer_agentic("air fryer", image=image, reranker="cross_encoder", session_id="s")
        router.assert_not_called()
        self.assertEqual(self.retrieve.call_count, 2)
        self.assertEqual(self.retrieve.call_args_list[1].args[0], "rewritten air fryer")
        for call in self.retrieve.call_args_list:
            self.assertIs(call.kwargs["image"], image)
            self.assertIsNone(call.kwargs["reranker"])
        self.assertIsNone(result["reranker"])

    def test_image_multipart_api_and_text_followups(self):
        from fastapi.testclient import TestClient
        from PIL import Image
        fake = types.ModuleType("search_core")
        fake.search = Mock(return_value=SOURCES)
        fake.available_components = lambda: ["bm25", "dense"]
        fake.available_rerankers = lambda: ["cross_encoder"]
        fake.get_product = self.products.get
        spec = importlib.util.spec_from_file_location("app_image_offline_test",
                 Path(__file__).resolve().parents[1] / "app.py")
        api = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"search_core": fake, "rag_core": self.rag}):
            spec.loader.exec_module(api)
        payload = io.BytesIO()
        Image.new("RGB", (2, 2), "white").save(payload, format="PNG")
        with TestClient(api.app) as client:
            for agentic in [False, True]:
                for text in ["air fryer", ""]:
                    with self.subTest(agentic=agentic, text=text):
                        self.rag.clear_history("photo")
                        self.retrieve.return_value = SOURCES[:2]
                        self.llm.return_value = "One recommendation [2]."
                        with patch.object(self.rag, "judge_sufficiency", return_value="YES"):
                            response = client.post("/api/v1/ask_image", data={
                                "q": text, "session_id": "photo", "agentic": str(agentic).lower(),
                                "reranker": "cross_encoder"},
                                files={"file": ("product.png", payload.getvalue(), "image/png")})
                        self.assertEqual(response.status_code, 200)
                        body = response.json()
                        self.assertEqual(body["query_mode"], "image+text" if text else "image")
                        self.assertEqual(body["focused_product_id"], 1)
                        self.assertIsNone(body["reranker"])
                        self.assertIsNotNone(self.retrieve.call_args.kwargs["image"])
                        response = client.get("/api/v1/ask", params={
                            "q": "How much capacity?", "session_id": "photo", "agentic": agentic})
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json()["sources"][0]["id"], 1)
                        self.assertEqual(response.json()["product_options"], SOURCES[:2])
                        # A new upload replaces old image choices and focus.
                        self.retrieve.return_value = SOURCES[3:5]
                        self.llm.return_value = "New choices [1] and [2]."
                        with patch.object(self.rag, "contextualize_query", return_value="thermostat"), \
                             patch.object(self.rag, "judge_sufficiency", return_value="YES"):
                            response = client.post("/api/v1/ask_image", data={
                                "q": "smart thermostat", "session_id": "photo", "agentic": str(agentic).lower()},
                                files={"file": ("other.png", payload.getvalue(), "image/png")})
                        self.assertEqual(response.status_code, 200)
                        self.assertIsNone(response.json()["focused_product_id"])
                        response = client.get("/api/v1/ask", params={
                            "q": "Weight of product 2?", "session_id": "photo", "agentic": agentic})
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json()["sources"][0]["id"], 4)
                        self.assertEqual(response.json()["product_options"], SOURCES[3:5])

    def test_api_selection_baseline_and_reset(self):
        from fastapi.testclient import TestClient
        fake = types.ModuleType("search_core")
        fake.search = Mock(return_value=SOURCES)
        fake.available_components = lambda: ["bm25", "dense"]
        fake.available_rerankers = lambda: []
        fake.get_product = self.products.get
        spec = importlib.util.spec_from_file_location("app_offline_test",
                 Path(__file__).resolve().parents[1] / "app.py")
        api = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"search_core": fake, "rag_core": self.rag}):
            spec.loader.exec_module(api)
        self.seed()
        with TestClient(api.app) as client:
            for agentic in [False, True]:
                response = client.get("/api/v1/ask", params={
                    "q": "Cái này chất liệu gì?", "session_id": "s",
                    "selected_product_id": 0, "agentic": agentic})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["sources"][0]["id"], 0)
                self.assertEqual(response.json()["intent"], "detail")
                for query, expected_id in [("Chi tiết cái 2", 1),
                                           ("Chất liệu gì?", 1),
                                           ("Chi tiết cái 1", 0)]:
                    response = client.get("/api/v1/ask", params={
                        "q": query, "session_id": "s", "agentic": agentic})
                    self.assertEqual(response.status_code, 200)
                    body = response.json()
                    self.assertEqual(body["intent"], "detail")
                    self.assertEqual(body["sources"][0]["id"], expected_id)
                    self.assertEqual(body["product_options"], SOURCES)
                self.llm.return_value = "I recommend only [2]."
                with patch.object(self.rag, "contextualize_query", return_value="coffee maker"), \
                     patch.object(self.rag, "judge_sufficiency", return_value="YES"):
                    response = client.get("/api/v1/ask", params={
                        "q": "Recommend a coffee maker", "session_id": "auto", "agentic": agentic})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["focused_product_id"], 1)
                response = client.get("/api/v1/ask", params={
                    "q": "How much capacity?", "session_id": "auto", "agentic": agentic})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["intent"], "detail")
                self.assertEqual(response.json()["sources"][0]["id"], 1)
                self.assertEqual(response.json()["product_options"], SOURCES)
                self.llm.return_value = "Weight: 2 kg [1]."
            self.assertEqual(client.get("/api/v1/ask", params={
                "q": "details", "selected_product_id": -1}).status_code, 422)
            with patch.object(self.rag, "contextualize_query", return_value="tent"):
                response = client.get("/api/v1/ask", params={
                    "q": "Product 2 weight?", "session_id": "s", "product_dialogue": False})
            self.assertEqual(response.json()["intent"], "search")
            response = client.post("/api/v1/reset_session", params={"session_id": "s"})
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("s", self.rag._PRODUCT_STATE)


if __name__ == "__main__":
    unittest.main()
