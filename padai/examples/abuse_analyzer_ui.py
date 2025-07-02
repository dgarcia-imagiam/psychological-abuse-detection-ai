import padai.config.bootstrap  # noqa: F401 always first import in main entry points

import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from padai.config.settings import settings
from padai.prompts.psychological_abuse import abuse_analyzer_prompts, abuse_analyzer_prompts_with_context
from padai.datasets.psychological_abuse import get_communications_df, get_communications_sample
from typing import Dict, Any
from padai.llms.base import ChatModelDescriptionEx, get_chat_model
from padai.llms.available import default_available_models_registry, default_available_models
import logging
from padai.chains.abuse_analyzer import get_abuse_analyzer_params
from padai.utils.text import make_label, strip_text

logger = logging.getLogger(__name__)


def build_chain(
    *,
    model_description: ChatModelDescriptionEx,
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

    llm = get_chat_model(model_description.engine, params)
    parser = StrOutputParser()

    return prompt | llm | parser


PRESET_LABELS = {
    "neutral": "Neutro",
    "vigilant": "Vigilante",
    "neutral_with_history": "Neutro (con antecedentes)",
    "extreme_vigilant_with_history": "Muy vigilante (con antecedentes)",
}

PRESET_PROMPTS_NO_CTX = abuse_analyzer_prompts[settings.language]["system"]
PRESET_PROMPTS_CTX = abuse_analyzer_prompts_with_context[settings.language]["system"]

DEFAULT_SYSTEM_NO_CTX = PRESET_PROMPTS_NO_CTX["neutral"]
DEFAULT_SYSTEM_CTX = PRESET_PROMPTS_CTX["neutral"]

DEFAULT_HUMAN_NO_CTX = abuse_analyzer_prompts[settings.language]["human"]["default"]
DEFAULT_HUMAN_CTX = abuse_analyzer_prompts_with_context[settings.language]["human"]["default"]

PREDEFINED_MESSAGES = get_communications_df()
DEFAULT_TEXT, DEFAULT_CONTEXT = get_communications_sample(PREDEFINED_MESSAGES, language=settings.language)


def _get_context_tab(context: str):
    return "ctx" if strip_text(context) else "no_ctx"


def _get_predefined_options():
    df = PREDEFINED_MESSAGES[PREDEFINED_MESSAGES["language"] == settings.language.value]

    return [
        {
            "label": make_label(row["text"], width=200),
            "value": idx
        }
        for idx, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Abuse Analyzer GUI", external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    # sticky="top",
    className="shadow-sm",
)

config_section = [
    dbc.Label("Modelo", html_for="model"),
    dcc.Dropdown(
        id="model",
        options=[{"label": m.label, "value": m.full_name} for m in default_available_models],
        value=default_available_models[0].id,
        clearable=False,
        maxHeight=600,
    ),

    html.Hr(className="my-3"),

    dbc.Label("Temperature", html_for="temperature"),
    dcc.Slider(
        id="temperature",
        min=0,
        max=1,
        step=0.05,
        value=0.0,
        marks={0: "0", 0.5: "0.5", 1: "1"},
    ),

    html.Hr(className="my-3"),

    dbc.Label("Preset de System prompt", html_for="system-preset"),
    dcc.Dropdown(
        id="system-preset",
        options=[{"label": PRESET_LABELS[k], "value": k} for k in PRESET_PROMPTS_NO_CTX],
        placeholder="Elegir preset…",
        clearable=True,
    ),

    dbc.Tabs(
        id="prompt-tabs",
        active_tab=_get_context_tab(DEFAULT_CONTEXT),
        className="mt-3",
        children=[
            dbc.Tab(
                label="Sin contexto",
                tab_id="no_ctx",
                children=[
                    dbc.Label("System prompt",
                              html_for="system-prompt-no-ctx"),
                    dcc.Textarea(
                        id="system-prompt-no-ctx",
                        value=DEFAULT_SYSTEM_NO_CTX,
                        rows=10,
                        style={"width": "100%"},
                        className="mb-3",
                    ),

                    dbc.Label("Human prompt",
                              html_for="human-prompt-no-ctx"),
                    dcc.Textarea(
                        id="human-prompt-no-ctx",
                        value=DEFAULT_HUMAN_NO_CTX,
                        rows=6,
                        style={"width": "100%"},
                    ),
                ],
            ),
            dbc.Tab(
                label="Con contexto",
                tab_id="ctx",
                children=[
                    dbc.Label("System prompt",
                              html_for="system-prompt-ctx"),
                    dcc.Textarea(
                        id="system-prompt-ctx",
                        value=DEFAULT_SYSTEM_CTX,
                        rows=20,
                        style={"width": "100%"},
                        className="mb-3",
                    ),

                    dbc.Label("Human prompt",
                              html_for="human-prompt-ctx"),
                    dcc.Textarea(
                        id="human-prompt-ctx",
                        value=DEFAULT_HUMAN_CTX,
                        rows=9,
                        style={"width": "100%"},
                    ),
                ],
            ),
        ],
    ),
]

sidebar = dbc.Card(
    [
        dbc.CardHeader(html.H5("Configuración")),
        dbc.CardBody(config_section)
    ],
    className="shadow-sm",
)

main_cards = html.Div(
    [
        dbc.Card(
            [
                dbc.CardHeader(html.H5("Mensaje a analizar y Contexto")),
                dbc.CardBody(
                    [
                        dbc.Label("Mensajes predefinidos", html_for="predefined-message"),
                        dcc.Dropdown(
                            id="predefined-message",
                            options=_get_predefined_options(),
                            placeholder="Elegir mensaje…",
                            clearable=True,
                            className="mb-3"
                        ),

                        dcc.Textarea(
                            id="user-input",
                            value=DEFAULT_TEXT,
                            rows=14,
                            style={"width": "100%"},
                            className="mb-3",
                        ),

                        dbc.Label("Contexto (opcional)", html_for="user-context"),
                        dcc.Textarea(
                            id="user-context",
                            value=DEFAULT_CONTEXT,
                            rows=6,
                            style={"width": "100%"},
                            className="mb-3",
                        ),

                        html.Div(
                            [
                                html.Span(id="error",
                                          className="text-danger fw-semibold me-2"),
                                dbc.Button("Analizar", id="analyze-btn",
                                           color="primary", className="ms-auto"),
                            ],
                            className="d-flex align-items-center",
                        ),
                    ],
                    className="p-3",
                ),
            ],
            className="shadow-sm mb-4",
        ),

        dbc.Card(
            [
                dbc.CardHeader(html.H5("Análisis generado")),
                dbc.CardBody(dcc.Markdown(id="output-md"), className="p-3"),
            ],
            style={"minHeight": "35vh", "overflowY": "auto"},
            className="shadow-sm",
        ),
    ],
)

app.layout = html.Div(
    [
        header,
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(sidebar, width=3),
                    dbc.Col(main_cards, width=9, md=9, lg=9),
                ],
                className="gy-4",
            ),
            fluid=True,
            className="pt-4",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

NO_TEMP_MODELS = {m.id for m in default_available_models if "no-temperature" in m.tags}


@app.callback(Output("temperature", "disabled"), Input("model", "value"))
def toggle_temp(model_id):
    if model_id is None:
        return no_update

    return model_id in NO_TEMP_MODELS


@app.callback(
    Output("prompt-tabs", "active_tab"),
    Input("user-context", "value"),
    State("prompt-tabs", "active_tab"),
    prevent_initial_call=True,
)
def flip_tab_on_context_change(user_ctx, current_tab):
    desired_tab = _get_context_tab(user_ctx)

    if desired_tab == current_tab:
        return no_update

    return desired_tab


@app.callback(
    [
        Output("user-input", "value"),
        Output("user-context", "value")
    ],
    Input("predefined-message", "value"),
    prevent_initial_call=True,
)
def change_predefined_message(predefined):
    if not predefined:
        return no_update, no_update

    try:
        user_input = PREDEFINED_MESSAGES.at[predefined, "text"]
        user_context = PREDEFINED_MESSAGES.at[predefined, "context"] or ""
        return user_input, user_context
    except Exception:
        return no_update, no_update


@app.callback(
    [
        Output("system-prompt-no-ctx", "value"),
        Output("system-prompt-ctx", "value")
    ],
    Input("system-preset", "value"),
    prevent_initial_call=True,
)
def apply_preset(preset):
    if not preset:
        return no_update, no_update

    try:
        sys_no_ctx = PRESET_PROMPTS_NO_CTX[preset]
        sys_ctx = PRESET_PROMPTS_CTX[preset]
        return sys_no_ctx, sys_ctx
    except KeyError:
        return no_update, no_update


@app.callback(
    Output("output-md", "children"),
    Output("error", "children"),
    Input("analyze-btn", "n_clicks"),
    State("model", "value"),
    State("temperature", "value"),
    State("system-prompt-no-ctx", "value"),
    State("human-prompt-no-ctx", "value"),
    State("system-prompt-ctx", "value"),
    State("human-prompt-ctx", "value"),
    State("user-input", "value"),
    State("user-context", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, model_id, temperature, system_prompt_no_ctx, human_prompt_no_ctx, system_prompt_ctx, human_prompt_ctx, user_input, user_context):
    if not user_input or not user_input.strip():
        return no_update, "Escribe un mensaje y pulsa Analizar."

    temp = None if model_id in NO_TEMP_MODELS else float(temperature or 0.0)

    model_description = default_available_models_registry[model_id]

    logger.info(
        "run_analysis: model_description=%s temperature=%s",
        model_description, temp
    )

    user_context = strip_text(user_context)
    params: Dict[str, str] = get_abuse_analyzer_params(user_input, user_context=user_context)

    if user_context:
        system_prompt = system_prompt_ctx
        human_prompt = human_prompt_ctx
    else:
        system_prompt = system_prompt_no_ctx
        human_prompt = human_prompt_no_ctx

    chain = build_chain(
        model_description=model_description,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        temperature=temp,
    )
    try:
        result: str = chain.invoke(params)
    except Exception as exc:  # fallback
        return "", f"⚠️ Error inesperado: {exc}"

    return result, ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
