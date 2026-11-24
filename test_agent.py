import json,unittest
from types import SimpleNamespace as S
from agent import run_agent
class AgentTest(unittest.TestCase):
 def test_retrieves_and_cites(self):
  call=S(id='1',function=S(name='search_reports',arguments=json.dumps({'query':'fastball'})))
  messages=iter([S(choices=[S(message=S(tool_calls=[call]))]),S(choices=[S(message=S(tool_calls=[],content='Fastball [report].'))])])
  class Completion:
   def create(self,**kwargs): return next(messages)
  client=S(chat=S(completions=Completion()))
  result=run_agent(client,'model','What pitch?',lambda q:[{'source':'report','text':'fastball'}],lambda:[])
  self.assertEqual(result['tools_used'],['search_reports']);self.assertEqual(result['sources'][0]['source'],'report')
 def test_does_not_answer_without_tool(self):
  class Completion:
   def create(self,**kwargs): return S(choices=[S(message=S(tool_calls=[],content='made up'))])
  result=run_agent(S(chat=S(completions=Completion())),'model','question',lambda q:[],lambda:[])
  self.assertNotEqual(result['answer'],'made up')
