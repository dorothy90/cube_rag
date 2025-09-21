#%%
from chroma_setup import main as index, query

# SQL
from chroma_setup import main as index

index(json_path="/Users/daehwankim/cube_rag/data/chunked_qa_pairs_sql.json",
 persist_dir="/Users/daehwankim/cube_rag/chroma_db",
      collection_name="qa_questions_sql")

      

index(json_path="/Users/daehwankim/cube_rag/data/chunked_qa_pairs_semiconductor.json",
persist_dir="/Users/daehwankim/cube_rag/chroma_db",
      collection_name="qa_questions_semiconductor")

index(json_path="/Users/daehwankim/cube_rag/data/chunked_qa_pairs_python.json",
persist_dir="/Users/daehwankim/cube_rag/chroma_db",
      collection_name="qa_questions_python")
#%%
# 존재 확인
query("chroma_db", "정규화와 비정규화의 차이점은?", k=3,
      collection_name="qa_questions_sql")
# %%
