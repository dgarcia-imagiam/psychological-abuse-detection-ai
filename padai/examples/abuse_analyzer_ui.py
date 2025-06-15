import padai.config.bootstrap  # noqa: F401 always first import in main entry points

import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from padai.config.settings import settings
from padai.prompts.psychological_abuse import abuse_analyzer_prompts
from padai.datasets.psychological_abuse import get_communications_df, get_communications_sample
from typing import Dict, Any, List, Set
from padai.llms.base import ChatModelDescription, get_chat_model
from pydantic import ConfigDict, Field
import logging

logger = logging.getLogger(__name__)


class AbuseChatModelDescription(ChatModelDescription):
    id: str
    label: str
    tags: Set[str] = Field(default_factory=set)

    model_config = ConfigDict(extra="forbid")


models: List[AbuseChatModelDescription] = [
    AbuseChatModelDescription(id="o3-pro",          label="OpenAI: o3-pro",             engine="openai",    params={"model": "o3-pro"},             tags={"no-temperature", "use-responses-api", }),
    AbuseChatModelDescription(id="o3",              label="OpenAI: o3",                 engine="openai",    params={"model": "o3"},                 tags={"no-temperature", }),
    AbuseChatModelDescription(id="o3-mini",         label="OpenAI: o3-mini",            engine="openai",    params={"model": "o3-mini"},            tags={"no-temperature", }),
    AbuseChatModelDescription(id="o4-mini",         label="OpenAI: o4-mini",            engine="openai",    params={"model": "o4-mini"},            tags={"no-temperature", }),
    AbuseChatModelDescription(id="gpt-4.5-preview", label="OpenAI: gpt-4.5-preview",    engine="openai",    params={"model": "gpt-4.5-preview"}),
    AbuseChatModelDescription(id="gpt-4.1",         label="OpenAI: gpt-4.1",            engine="openai",    params={"model": "gpt-4.1"}),
    AbuseChatModelDescription(id="gpt-4.1-mini",    label="OpenAI: gpt-4.1-mini",       engine="openai",    params={"model": "gpt-4.1-mini"}),
    AbuseChatModelDescription(id="gpt-4.1-nano",    label="OpenAI: gpt-4.1-nano",       engine="openai",    params={"model": "gpt-4.1-nano"}),
    AbuseChatModelDescription(id="gpt-4o",          label="OpenAI: gpt-4o",             engine="openai",    params={"model": "gpt-4o"}),
    AbuseChatModelDescription(id="gpt-4o-mini",     label="OpenAI: gpt-4o-mini",        engine="openai",    params={"model": "gpt-4o-mini"}),

    AbuseChatModelDescription(id="nova-premier",    label="AWS BedRock: Nova Premier",  engine="bedrock",   params={"model": "us.amazon.nova-premier-v1:0", "region_name": "us-east-1"}),
    AbuseChatModelDescription(id="nova-pro",        label="AWS BedRock: Nova Pro",      engine="bedrock",   params={"model": "amazon.nova-pro-v1:0", "region_name": "us-east-1"}),
    AbuseChatModelDescription(id="nova-lite",       label="AWS BedRock: Nova Lite",     engine="bedrock",   params={"model": "amazon.nova-lite-v1:0", "region_name": "us-east-1"}),
    AbuseChatModelDescription(id="nova-micro",      label="AWS BedRock: Nova Micro",    engine="bedrock",   params={"model": "amazon.nova-micro-v1:0", "region_name": "us-east-1"}),
    AbuseChatModelDescription(id="deep-seek",       label="AWS BedRock: DeepSeek",      engine="bedrock",   params={"model": "us.deepseek.r1-v1:0", "region_name": "us-west-2"}),
]

models_registry: Dict[str, AbuseChatModelDescription] = {
    m.id: m for m in models
}


def build_chain(
    *,
    model_description: AbuseChatModelDescription,
    system_prompt: str,
    human_prompt: str,
    temperature: float | None,
):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    params: Dict[str, Any] = model_description.params.copy()

    if "no-temperature" not in model_description.tags:
        params["temperature"] = temperature

    if "use-responses-api" in model_description.tags:
        params["use_responses_api"] = True

    llm = get_chat_model(model_description.engine, params)
    parser = StrOutputParser()

    return prompt | llm | parser


PRESET_PROMPTS = abuse_analyzer_prompts[settings.language]["system"]

PRESET_LABELS = {
    "neutral": "Neutro",
    "vigilant": "Vigilante",
    "neutral_with_history": "Neutro (con antecedentes)",
    "extreme_vigilant_with_history": "Muy vigilante (con antecedentes)",
}

DEFAULT_SYSTEM = PRESET_PROMPTS["neutral"]

DEFAULT_HUMAN = abuse_analyzer_prompts[settings.language]["human"]["default"]

DEFAULT_TEXT, _ = get_communications_sample(get_communications_df(), language=settings.language)


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Abuse Analyzer GUI", external_stylesheets=[dbc.themes.BOOTSTRAP])

sidebar = dbc.Card(
    [
        dbc.CardHeader(html.H5("Configuración")),
        dbc.CardBody(
            [
                dbc.Label("Modelo"),
                dcc.Dropdown(id="model", options=[{"label": m.label, "value": m.id} for m in models],
                             value=models[0].id, clearable=False, maxHeight=600),

                html.Hr(className="my-3"),

                dbc.Label("Temperature"),
                dcc.Slider(id="temperature", min=0, max=1,
                           step=0.05, value=0.0,
                           marks={0: "0", 0.5: "0.5", 1: "1"}),

                html.Hr(className="my-3"),

                dbc.Label("Preset de System prompt"),
                dcc.Dropdown(
                    id="system-preset",
                    options=[
                        {
                            "label": PRESET_LABELS[key],
                            "value": key
                        }
                        for key in PRESET_PROMPTS
                    ],
                    placeholder="Elegir preset…",
                    clearable=True,
                ),
                html.Br(),

                dbc.Label("System prompt"),
                dcc.Textarea(id="system-prompt", value=DEFAULT_SYSTEM, rows=12, style={"width": "100%"}),

                html.Br(),

                dbc.Label("Human prompt"),

                dcc.Textarea(id="human-prompt", value=DEFAULT_HUMAN, rows=8, style={"width": "100%"}),
            ]
        ),
    ],
    style={"height": "100vh", "overflowY": "auto"},
    className="shadow-sm",
)

header = dbc.Navbar(
    dbc.Container(
        dbc.NavbarBrand(
            "Analizador de maltrato psicológico",
            className="mx-auto fw-semibold",
        ),
        fluid=True,
    ),
    color="primary",
    dark=True,
    sticky="top",
    className="shadow-sm",
)

app.layout = html.Div(
    [
        header,
        dbc.Container(
            [
                dbc.Row(
                    [
                        # --- Sidebar column (left) ------------------------
                        dbc.Col(sidebar, width=3),

                        # --- Main column (right) -------------------------
                        dbc.Col(
                            [
                                dbc.Col(          # centred inner column
                                    [
                                        dbc.Label("Mensaje a analizar"),
                                        dcc.Textarea(
                                            id="user-input",
                                            value=DEFAULT_TEXT,
                                            rows=16,
                                            style={"width": "100%"},
                                        ),
                                        html.Br(),

                                        html.Div(
                                            [
                                                html.Span(
                                                    id="error",
                                                    className="text-danger fw-semibold me-2",
                                                ),
                                                dbc.Button(
                                                    "Analizar",
                                                    id="analyze-btn",
                                                    color="primary",
                                                    className="ms-auto",
                                                ),
                                            ],
                                            className="d-flex align-items-center",
                                        ),
                                        html.Br(),

                                        dbc.Label("Análisis generado"),
                                        dbc.Card(
                                            dbc.CardBody(
                                                dcc.Markdown(id="output-md"),
                                                className="p-3",
                                            ),
                                            style={
                                                "minHeight": "260px",
                                                "overflowY": "auto",
                                            },
                                            className="shadow-sm",
                                        ),
                                    ],
                                    width=12, md=10, lg=8, xl=7,
                                    className="mx-auto",
                                ),
                            ],
                            width=9,
                        ),
                    ],
                    className="gy-4 align-items-start",
                ),
            ],
            fluid=True,
            className="pt-4",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

NO_TEMP_MODELS = {m.id for m in models if "no-temperature" in m.tags}


@app.callback(Output("temperature", "disabled"), Input("model", "value"))
def toggle_temp(model_id):
    if model_id is None:
        return no_update

    return model_id in NO_TEMP_MODELS


@app.callback(
    Output("system-prompt", "value"),
    Input("system-preset", "value"),
    prevent_initial_call=True,
)
def _apply_preset(preset):
    if preset and preset in PRESET_PROMPTS:
        return PRESET_PROMPTS[preset]
    return no_update


@app.callback(
    Output("output-md", "children"),
    Output("error", "children"),
    Input("analyze-btn", "n_clicks"),
    State("model", "value"),
    State("temperature", "value"),
    State("system-prompt", "value"),
    State("human-prompt", "value"),
    State("user-input", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, model_id, temperature, system_prompt, human_prompt, user_msg):
    if not user_msg or not user_msg.strip():
        return no_update, "Escribe un mensaje y pulsa Analizar."

    temp = None if model_id in NO_TEMP_MODELS else float(temperature or 0.0)

    model_description = models_registry[model_id]

    logger.info(
        "run_analysis: model_description=%s temperature=%s",
        model_description, temp
    )

    chain = build_chain(
        model_description=model_description,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        temperature=temp,
    )
    try:
        result: str = chain.invoke({"user_input": user_msg})
    except Exception as exc:  # fallback
        return "", f"⚠️ Error inesperado: {exc}"

    return result, ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
