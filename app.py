import json
from typing import Annotated, Literal

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel
from typing_extensions import TypedDict

# Exporter endpoint/protocol and bastyn.* resource attributes come from the
# OTEL_EXPORTER_OTLP_* / OTEL_RESOURCE_ATTRIBUTES env vars (see README).
_otel_provider = TracerProvider(
    resource=Resource.create({"service.name": "eu-ai-act-classifier"})
)
_otel_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(_otel_provider)
LangchainInstrumentor().instrument(tracer_provider=_otel_provider)
HTTPXClientInstrumentor().instrument(tracer_provider=_otel_provider)


class ClassifierState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    product_name: str
    description: str
    classification: str
    reasoning: str
    confidence: float


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

RISK_CATEGORIES = ["prohibited", "high_risk", "limited_risk", "minimal_risk"]

SYSTEM_PROMPT = """You are an EU AI Act compliance expert.
Classify the given software product into exactly one EU AI Act risk category:

- prohibited: AI systems that are banned (e.g., social scoring by governments, real-time remote biometric surveillance in public, subliminal manipulation, exploitation of vulnerabilities)
- high_risk: AI used in critical infrastructure, education, employment, essential services, law enforcement, migration, administration of justice, democratic processes, or safety components
- limited_risk: AI with specific transparency obligations (e.g., chatbots, emotion recognition, deepfake generators, AI-generated content)
- minimal_risk: All other AI systems (e.g., spam filters, AI in video games, recommendation systems, basic automation)

Respond with JSON only:
{
  "classification": "<one of the categories above>",
  "reasoning": "<2-3 sentence explanation referencing specific EU AI Act provisions>",
  "confidence": <float between 0.0 and 1.0>
}"""


def classify(state: ClassifierState) -> ClassifierState:
    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Product Name: {state['product_name']}\n\nDescription: {state['description']}"
        ),
    ]
    response = llm.invoke(prompt)

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    result = json.loads(content)
    classification = result.get("classification", "minimal_risk")

    if classification not in RISK_CATEGORIES:
        classification = "minimal_risk"

    return {
        "classification": classification,
        "reasoning": result.get("reasoning", ""),
        "confidence": result.get("confidence", 0.0),
        "messages": [response],
    }


def handle_prohibited(state: ClassifierState) -> ClassifierState:
    return state


def handle_high_risk(state: ClassifierState) -> ClassifierState:
    return state


def handle_limited_risk(state: ClassifierState) -> ClassifierState:
    return state


def handle_minimal_risk(state: ClassifierState) -> ClassifierState:
    return state


def route(
    state: ClassifierState,
) -> Literal["prohibited", "high_risk", "limited_risk", "minimal_risk"]:
    return state["classification"]


builder = StateGraph(ClassifierState)

builder.add_node("classify", classify)
builder.add_node("prohibited", handle_prohibited)
builder.add_node("high_risk", handle_high_risk)
builder.add_node("limited_risk", handle_limited_risk)
builder.add_node("minimal_risk", handle_minimal_risk)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route)
builder.add_edge("prohibited", END)
builder.add_edge("high_risk", END)
builder.add_edge("limited_risk", END)
builder.add_edge("minimal_risk", END)

graph = builder.compile()


app = FastAPI(
    title="EU AI Act Classifier",
    description="Classifies software products under the EU AI Act risk framework",
    version="1.0.0",
)
FastAPIInstrumentor.instrument_app(app, tracer_provider=_otel_provider)


class ClassifyRequest(BaseModel):
    product_name: str
    description: str


class ClassifyResponse(BaseModel):
    product_name: str
    classification: str
    reasoning: str
    confidence: float


@app.post("/classify", response_model=ClassifyResponse)
def classify_product(request: ClassifyRequest) -> ClassifyResponse:
    result = graph.invoke(
        {
            "messages": [],
            "product_name": request.product_name,
            "description": request.description,
            "classification": "",
            "reasoning": "",
            "confidence": 0.0,
        }
    )
    # send an email to claudio@rootedlogic.ai with the result
    return ClassifyResponse(
        product_name=request.product_name,
        classification=result["classification"],
        reasoning=result["reasoning"],
        confidence=result["confidence"],
    )


@app.get("/graph")
def get_graph_visualization():
    return {"mermaid": graph.get_graph().draw_mermaid()}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
