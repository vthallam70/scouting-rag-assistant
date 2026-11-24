"""RAG API: document indexing, semantic search, cited answer generation."""
import os
import psycopg
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from agent import run_agent
app=FastAPI(title='Scouting assistant')
MODEL='text-embedding-3-small'
def connect(): return psycopg.connect(os.environ['DATABASE_URL'])
def vector(v): return '['+','.join(str(float(x)) for x in v)+']'
def embed(text): return OpenAI().embeddings.create(model=MODEL,input=text).data[0].embedding
class Document(BaseModel): source:str=Field(min_length=1,max_length=300); text:str=Field(min_length=1,max_length=20000)
class Question(BaseModel): question:str=Field(min_length=1,max_length=2000)
@app.on_event('startup')
def setup():
 with connect() as db:
  db.execute('CREATE EXTENSION IF NOT EXISTS vector')
  db.execute('CREATE TABLE IF NOT EXISTS passages (id bigserial PRIMARY KEY, source text NOT NULL, body text NOT NULL, embedding vector(1536) NOT NULL)')
@app.post('/documents')
def add_document(doc:Document):
 chunks=[p.strip() for p in doc.text.split('\n\n') if p.strip()]
 with connect() as db:
  for chunk in chunks: db.execute('INSERT INTO passages(source,body,embedding) VALUES (%s,%s,%s::vector)',(doc.source,chunk,vector(embed(chunk))))
 return {'indexed':len(chunks)}
def search_reports(query):
 with connect() as db: rows=db.execute('SELECT source,body FROM passages ORDER BY embedding <=> %s::vector LIMIT 4',(vector(embed(query)),)).fetchall()
 return [{'source':source,'text':body} for source,body in rows]
def list_sources():
 with connect() as db: return [row[0] for row in db.execute('SELECT DISTINCT source FROM passages ORDER BY source LIMIT 100').fetchall()]
@app.post('/ask')
def ask(q:Question):
 return run_agent(OpenAI(),os.getenv('CHAT_MODEL','gpt-4o-mini'),q.question,search_reports,list_sources)
