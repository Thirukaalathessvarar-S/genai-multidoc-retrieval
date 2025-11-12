## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

### DESIGN STEPS:
#### STEP 1: Dataset Preparation
    Downloaded research papers in PDF format, such as MetaGPT, SWE-Bench, LongLoRA, and LoftQ.
    Mapped each paper to its respective retrieval and summarization tools using a utility function (get_doc_tools).

#### STEP 2: Tool Generation
    Created vector tools for document similarity analysis and summary tools for abstracting key content.
    Stored these tools for each document in a dictionary (paper_to_tools_dict).

#### STEP 3: Index Construction
    Used VectorStoreIndex and ObjectIndex from LlamaIndex to index the document tools.
    Configured a retriever to fetch the most relevant tools based on similarity to the query.

#### STEP 4: Query Handling
    Developed an AgentWorker to process queries and retrieve necessary tools using the configured retriever.
    Implemented a FunctionCallingAgentWorker to ensure all responses are tool-based without reliance on prior knowledge.

#### STEP 5: Evaluation
    Designed diverse queries for testing, such as comparing evaluation datasets used in MetaGPT and SWE-Bench, and analyzing approaches in LongLoRA and LoftQ.
    Synthesized concise and comparative responses using the agent system.


### PROGRAM:
```
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Tell me about the evaluation dataset used in LongLoRA, "
    "and then tell me about the evaluation results"
)

response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(response))

response = agent.query(
    "Tell me about the evaluation dataset used "
    "in MetaGPT and compare it against SWE-Bench"
)
print(str(response))
```

### OUTPUT:
<img width="1098" height="462" alt="389508509-7e8f628a-b364-47d6-b5d3-29d7ffc45215" src="https://github.com/user-attachments/assets/ab2ba081-fa79-4b9f-b91f-d5aae05e0b98" />

<img width="1063" height="517" alt="389508577-8f1f174c-68a4-4342-8bfd-95c8d0ffb5ad" src="https://github.com/user-attachments/assets/55dd8816-6fef-4f2e-a47a-2ca7f12d19f3" />

<img width="1058" height="528" alt="389508605-51268415-5721-4abc-87da-3a0dc54d4d59" src="https://github.com/user-attachments/assets/9edeed12-3c07-4137-b969-05a0fc1a7c3f" />



### RESULT:
The multidocument retrieval agent was successfully designed and implemented using LlamaIndex. The agent demonstrated its capability to retrieve and synthesize information from multiple academic papers, answering complex queries with concise, relevant, and accurate responses.
