import unittest
from assistant import search
class RetrievalTest(unittest.TestCase):
 def test_relevant_passage_first(self):
  docs=[{'source':'a','text':'pitcher throws fastballs'},{'source':'b','text':'outfielder catches fly balls'}]
  self.assertEqual(search(docs,'pitcher fastballs')[0][1]['source'],'a')
