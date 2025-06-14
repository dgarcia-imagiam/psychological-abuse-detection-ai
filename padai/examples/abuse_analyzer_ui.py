import padai.config.bootstrap  # noqa: F401 always first import in main entry points

import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import openai
from padai.config.settings import settings
from padai.prompts.psychological_abuse import abuse_analyzer_prompts
from padai.datasets.psychological_abuse import get_communications_df, get_communications_sample
from typing import Dict, Any


def build_chain(
    *,
    model_name: str,
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

    llm_kwargs: Dict[str, Any] = {"model": model_name}

    if temperature is not None:
        llm_kwargs["temperature"] = temperature

    if model_name in {"o3-pro", }:
        llm_kwargs["use_responses_api"] = True

    llm = ChatOpenAI(**llm_kwargs, api_key=settings.openai.api_key)
    parser = StrOutputParser()

    return prompt | llm | parser


NO_TEMPERATURE_MODELS = {"o3-pro", "o3", "o3-mini", "o4-mini"}

PRESET_PROMPTS = abuse_analyzer_prompts[settings.language]["system"]

PRESET_LABELS = {
    "neutral": "Neutro",
    "vigilant": "Vigilante",
    "neutral_with_history": "Neutro (con antecedentes)",
    "extreme_vigilant_with_history": "Muy vigilante (con antecedentes)",
}

DEFAULT_SYSTEM = PRESET_PROMPTS["neutral"]

DEFAULT_HUMAN = abuse_analyzer_prompts[settings.language]["human"]["default"]

MODEL_OPTIONS = [
    {"label": "o3-pro", "value": "o3-pro"},
    {"label": "o3", "value": "o3"},
    {"label": "o3-mini", "value": "o3-mini"},
    {"label": "o4-mini", "value": "o4-mini"},
    {"label": "gpt-4.5-preview", "value": "gpt-4.5-preview"},
    {"label": "gpt-4.1", "value": "gpt-4.1"},
    {"label": "gpt-4.1-mini", "value": "gpt-4.1-mini"},
    {"label": "gpt-4.1-nano", "value": "gpt-4.1-nano"},
    {"label": "gpt-4o", "value": "gpt-4o"},
    {"label": "gpt-4o-mini", "value": "gpt-4o-mini"},
]

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
                dcc.Dropdown(id="model", options=MODEL_OPTIONS,
                             value="o3-pro", clearable=False),

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

@app.callback(Output("temperature", "disabled"), Input("model", "value"))
def toggle_temp(model):
    return model in NO_TEMPERATURE_MODELS


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
def run_analysis(n_clicks, model, temperature, system_prompt, human_prompt, user_msg):
    if not user_msg or not user_msg.strip():
        return no_update, "Escribe un mensaje y pulsa Analizar."

    temp = None if model in NO_TEMPERATURE_MODELS else float(temperature or 0.0)
    chain = build_chain(model_name=model, system_prompt=system_prompt, human_prompt=human_prompt, temperature=temp)
    try:
        result: str = chain.invoke({"user_input": user_msg})
    except openai.OpenAIError as exc:
        return "", f"⚠️ Error: {exc}"
    except Exception as exc:  # fallback
        return "", f"⚠️ Error inesperado: {exc}"

    return result, ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
