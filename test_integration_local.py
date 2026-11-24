"""Free local integration test: real FastAPI routes + real Postgres/pgvector, no paid API calls.

The OpenAI embedding call is replaced by a deterministic hashed bag-of-words embedder, and the
chat model by a scripted client that issues a real `search_reports` tool call. Everything else
(chunking, vector storage, cosine search in pgvector, the agent's tool loop, citations) runs for real.

Run: DATABASE_URL=postgresql://... python -m unittest -v test_integration_local
Skips when DATABASE_URL is not set.
"""
import hashlib, json, math, os, re, unittest
from types import SimpleNamespace as S

DIM = 1536

def local_embed(text):
    v = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 20) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

class ScriptedLLM:
    """Round 1: call search_reports with the user question. Round 2: answer from the top passage."""
    def __init__(self):
        self.chat = S(completions=self)
    def create(self, model, messages, tools, temperature):
        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        if not tool_msgs:
            q = messages[1]["content"]
            call = S(id="c1", function=S(name="search_reports", arguments=json.dumps({"query": q})))
            return S(choices=[S(message=S(tool_calls=[call], content=None))])
        top = json.loads(tool_msgs[-1]["content"])[0]
        return S(choices=[S(message=S(tool_calls=[], content=f"{top['text']} [{top['source']}]"))])

REPORTS = {
    "report-pitcher-a": "Pitcher A throws a fastball early in the count.\n\nPitcher A leans on a slider with two strikes.",
    "report-hitter-b": "Hitter B chases high pitches and struggles against changeups.\n\nHitter B pulls ground balls to the left side.",
    "report-catcher-c": "Catcher C has a quick release and throws out runners at second base often.",
}
CASES = [
    ("What does Pitcher A throw early in the count?", "report-pitcher-a", "fastball"),
    ("Which pitch does Hitter B struggle against?", "report-hitter-b", "changeups"),
    ("How does Catcher C handle runners at second base?", "report-catcher-c", "release"),
]

@unittest.skipUnless(os.getenv("DATABASE_URL"), "needs DATABASE_URL for a local pgvector database")
class LocalRagIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rag_api
        from fastapi.testclient import TestClient
        rag_api.embed = local_embed
        rag_api.OpenAI = ScriptedLLM
        with rag_api.connect() as db:
            db.execute("CREATE EXTENSION IF NOT EXISTS vector")
            db.execute("DROP TABLE IF EXISTS passages")
        cls.client = TestClient(rag_api.app).__enter__()  # runs the startup hook (creates the table)
        for src, text in REPORTS.items():
            r = cls.client.post("/documents", json={"source": src, "text": text})
            assert r.status_code == 200, r.text

    def test_indexed_all_chunks(self):
        import rag_api
        with rag_api.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM passages").fetchone()[0], 5)

    def test_retrieval_and_cited_answers(self):
        for question, source, term in CASES:
            with self.subTest(question=question):
                r = self.client.post("/ask", json={"question": question})
                self.assertEqual(r.status_code, 200, r.text)
                body = r.json()
                self.assertEqual(body["tools_used"], ["search_reports"])
                self.assertEqual(body["sources"][0]["source"], source)
                self.assertIn(term, body["answer"].lower())
                self.assertIn(f"[{source}]", body["answer"])

    def test_rejects_empty_question(self):
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)

if __name__ == "__main__":
    unittest.main()
