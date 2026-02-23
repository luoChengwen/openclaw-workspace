# test openclaw

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# test openclaw

PROJECT_ROOT = Path(__file__).parent
CODE_DIR = PROJECT_ROOT / "code_generation"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_FILE = PROJECT_ROOT / "logs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)

# test openclaw

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {message}"
    print(log_msg)
    
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

# test openclaw

def log_agent(agent_name: str, input_data: Any, output_data: Any):
    log(f"\n{'='*50}")
    log(f"Agent: {agent_name}")
    log(f"{'='*50}")
    log(f"INPUT: {str(input_data)[:500]}")
    log(f"OUTPUT: {str(output_data)[:1000]}")

# test openclaw

def get_llm():
    # test openclaw
    try:
        from crewai_llm import CrewAILLM
        return CrewAILLM(
            temperature=0,
            max_completion_tokens=4096
        )
    except ImportError:
        from openai import OpenAI
        return OpenAI(api_key="your-api-key")

# test openclaw

from crewai.tools import BaseTool
from pydantic import Field
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import chromadb
import re

# test openclaw

class CodeExtractionTool(BaseTool):
    name: str = "code_extraction"
    description: str = "test openclaw"
    
    def _run(self, folder_path: str = None) -> str:
        if not folder_path:
            folder_path = str(CODE_DIR)
        
        path = Path(folder_path)
        if not path.exists():
            return json.dumps({"error": f"Folder not found: {folder_path}"})
        
        reader = SimpleDirectoryReader(
            str(path),
            required_exts=[".py", ".ipynb"]
        )
        documents = reader.load_data()
        
        result = {
            "files": [],
            "total_files": len(documents)
        }
        
        for doc in documents:
            result["files"].append({
                "file_name": doc.metadata.get("file_name", "unknown"),
                "content": doc.text[:3000],
                "file_size": len(doc.text)
            })
        
        return json.dumps(result, indent=2)

# test openclaw

class SchemaExtractionTool(BaseTool):
    name: str = "schema_extraction"
    description: str = "test openclaw"
    
    def _run(self, code_content: str = None) -> str:
        if not code_content:
            return json.dumps({"error": "test openclaw"})
        
        tables = set()
        table_patterns = [
            r'FROM\s+([a-zA-Z0-9_\.]+)',
            r'JOIN\s+([a-zA-Z0-9_\.]+)',
            r'TABLE\s+([a-zA-Z0-9_]+)',
            r'\.table\([\'"]([\w\.]+)[\'"]\)',
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, code_content, re.IGNORECASE)
            tables.update(matches)
        
        columns = set()
        col_patterns = [
            r'SELECT\s+(.+?)\s+FROM',
            r'AS\s+([a-zA-Z0-9_]+)',
        ]
        
        for pattern in col_patterns:
            matches = re.findall(pattern, code_content, re.IGNORECASE)
            columns.update([m.strip() for m in matches if m.strip()])
        
        return json.dumps({
            "tables": list(tables)[:50],
            "columns": list(columns)[:100],
            "table_count": len(tables),
            "column_count": len(columns)
        }, indent=2)

# test openclaw

class RAGQueryTool(BaseTool):
    name: str = "rag_query"
    description: str = "test openclaw"
    
    def _run(self, query: str, top_k: int = 3) -> str:
        chroma_client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
        
        try:
            collection = chroma_client.get_collection("code_knowledge")
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            return json.dumps({
                "query": query,
                "results": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0]
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e), "query": query})

# test openclaw

class SQLValidatorTool(BaseTool):
    name: str = "sql_validator"
    description: str = "test openclaw"
    
    def _run(self, sql: str = None) -> str:
        if not sql:
            return json.dumps({"error": "test openclaw"})
        
        issues = []
        
        dangerous_commands = ["DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE"]
        for cmd in dangerous_commands:
            if re.search(rf'\b{cmd}\b', sql, re.IGNORECASE):
                issues.append(f"Warning: Contains {cmd} command")
        
        if "SELECT" not in sql.upper():
            issues.append("Error: test openclaw")
        
        if not sql.strip().endswith(";"):
            issues.append("Warning: test openclaw")
        
        if sql.count("(") != sql.count(")"):
            issues.append("Error: test openclaw")
        
        return json.dumps({
            "valid": len([i for i in issues if i.startswith("Error")]) == 0,
            "issues": issues,
            "sql_length": len(sql)
        }, indent=2)

# test openclaw

def load_agents_from_yaml(yaml_path: str = "config/agent.yaml"):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    llm = get_llm()
    
    code_tool = CodeExtractionTool()
    schema_tool = SchemaExtractionTool()
    rag_tool = RAGQueryTool()
    validator_tool = SQLValidatorTool()
    
    agents = {}
    
    for name, params in config.items():
        if name == "query_generation_agent":
            continue
        if name == "code_validator_agent":
            continue
            
        agent = Agent(
            role=params.get("role", name),
            goal=params.get("goal", ""),
            backstory=params.get("backstory", ""),
            verbose=params.get("verbose", True),
            allow_delegation=params.get("allow_delegation", False),
            llm=llm
        )
        
        if "code" in name.lower() and "extraction" in name.lower():
            agent.tools = [code_tool]
        elif "schema" in name.lower():
            agent.tools = [schema_tool]
        elif "knowledge" in name.lower():
            agent.tools = [rag_tool, schema_tool]
        
        agents[name] = agent
    
    return agents, [code_tool, schema_tool, rag_tool, validator_tool]

# test openclaw

class AgentTester:
    
    def __init__(self):
        self.results = {}
        self.agents = None
        self.tools = None
    
    def load_agents(self):
        log("test openclaw")
        try:
            self.agents, self.tools = load_agents_from_yaml()
            log(f"test openclaw", "SUCCESS")
        except Exception as e:
            log(f"test openclaw: {e}", "ERROR")
            raise
    
    def test_single_agent(self, agent_name: str, input_data: str):
        log(f"\ntest openclaw: {agent_name}")
        
        if agent_name not in self.agents:
            return {"error": f"test openclaw"}
        
        agent = self.agents[agent_name]
        
        task = Task(
            description=input_data,
            agent=agent,
            expected_output="test openclaw"
        )
        
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        self.results[agent_name] = {
            "input": input_data,
            "output": str(result),
            "status": "success"
        }
        
        log(f"test openclaw", "SUCCESS")
        return self.results[agent_name]
    
    def test_full_pipeline(self, user_request: str):
        log(f"\n{'='*60}")
        log(f"test openclaw")
        log(f"test openclaw: {user_request}")
        log(f"{'='*60}")
        
        results = {}
        
        log("test openclaw")
        code_result = self.test_single_agent(
            "Code Extraction Specialist",
            f"test openclaw"
        )
        results["step1_code"] = code_result
        log_agent("Code Extraction", user_request, code_result)
        
        log("test openclaw")
        schema_result = self.test_single_agent(
            "Schema_read_agent",
            code_result.get("output", "")[:2000]
        )
        results["step2_schema"] = schema_result
        log_agent("Schema Extraction", code_result, schema_result)
        
        log("test openclaw")
        knowledge_result = self.test_single_agent(
            "Knowledge Layer Builder",
            schema_result.get("output", "")
        )
        results["step3_knowledge"] = knowledge_result
        log_agent("Knowledge Layer", schema_result, knowledge_result)
        
        log("test openclaw")
        sql_result = f"test openclaw"
        results["step4_sql"] = sql_result
        log_agent("SQL Generation", user_request, sql_result)
        
        log("test openclaw")
        validator = self.tools[3]
        validation = validator._run(sql_result)
        results["step5_validation"] = validation
        log_agent("SQL Validation", sql_result, validation)
        
        output_file = OUTPUT_DIR / f"pipeline_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        log(f"test openclaw", "SUCCESS")
        
        return results
    
    def run_tests(self):
        log("test openclaw")
        
        log("\n" + "="*50)
        log("test openclaw")
        log("="*50)
        self.test_single_agent(
            "Code Extraction Specialist",
            str(CODE_DIR)
        )
        
        log("\n" + "="*50)
        log("test openclaw")
        log("="*50)
        self.test_full_pipeline(
            "test openclaw"
        )
        
        log("\n" + "="*50)
        log("test openclaw")
        log("="*50)
        
        print("\ntest openclaw:")
        for agent_name, result in self.results.items():
            status = result.get("status", "unknown")
            print(f"  - {agent_name}: {status}")

# test openclaw

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║         Project Agent - Testing & Validation              ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) < 2:
        print("test openclaw")
        print("  python test_agents.py test")
        print("  python test_agents.py run <query>")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "test":
        tester = AgentTester()
        tester.load_agents()
        tester.run_tests()
        
    elif mode == "run":
        query = " ".join(sys.argv[2:])
        log(f"test openclaw: {query}")
        
        tester = AgentTester()
        tester.load_agents()
        result = tester.test_full_pipeline(query)
        
        print("\ntest openclaw:")
        print(result)
    
    else:
        print(f"test openclaw: {mode}")

# test openclaw

if __name__ == "__main__":
    main()
