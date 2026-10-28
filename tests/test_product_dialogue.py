"""Offline regressions: no embedding downloads, catalog writes, or Ollama calls.

Run: python -m unittest discover -s tests -v
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from product_dialogue import route_product_query


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
                self.retrieve.assert_not_called()

    def test_focus_survives_multiple_detail_turns(self):
        self.seed()
        self.rag.answer("Product 2 weight?", session_id="s")
        for _ in range(4):
            result = self.rag.answer("Is it waterproof?", session_id="s")
            self.assertEqual(result["sources"][0]["id"], 1)
        self.assertEqual(len(self.rag.get_history("s")), 3)

    def test_selection_zero_and_older_card(self):
        self.seed()
        self.rag.answer("Product 5 weight?", session_id="s")
        result = self.rag.answer("Cái này chất liệu gì?", session_id="s", selected_product_id=0)
        self.assertEqual(result["sources"][0]["id"], 0)

    def test_clarify_preserves_options_and_does_not_call_llm(self):
        self.seed()
        result = self.rag.answer("Cái đó chất liệu gì?", session_id="s")
        self.assertEqual(result["intent"], "clarify")
        self.assertEqual(len(result["sources"]), 5)
        self.llm.assert_not_called()
        result = self.rag.answer("Product 4 details?", session_id="s")
        self.assertEqual(result["sources"][0]["id"], 3)

    def test_new_search_clears_focus(self):
        self.seed()
        with patch.object(self.rag, "contextualize_query", return_value="coffee maker"):
            result = self.rag.answer("Recommend a coffee maker", session_id="s", selected_product_id=0)
        self.assertEqual(result["intent"], "search")
        self.retrieve.assert_called_once()
        self.assertIsNone(self.rag._PRODUCT_STATE["s"]["focused_id"])

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
