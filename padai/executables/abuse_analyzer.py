from __future__ import annotations

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_chain(model_name: str = "gpt-4.5-preview", temperature: float = 0.0):
    prompt = ChatPromptTemplate.from_messages(
        [
            #("system", "Eres un experto en psicología forense que analiza mensajes escritos para detectar abusos psicológicos. Cuando el usuario te pida analizar un mensaje, identifica de manera objetiva y neutral cualquier signo explícito o implícito de abuso o manipulación. Analiza el tono del mensaje, el lenguaje (insultos, amenazas), los intentos de control o coacción, la culpabilización, el gaslighting y cualquier otra estrategia tóxica que detectes, incluso si no encaja en las categorías anteriores. Presenta un análisis exhaustivo en español, explicando cada conclusión clave en un texto claro y coherente."),
            #("human", "Identifica de manera objetiva y neutral cualquier signo explícito o implícito de abuso o manipulación en el mensaje copiado más adelante. Analiza el tono del mensaje, el lenguaje (insultos, amenazas), los intentos de control o coacción, la culpabilización, el gaslighting y cualquier otra estrategia tóxica que detectes, incluso si no encaja en las categorías anteriores. Presenta un análisis exhaustivo en español, explicando cada conclusión clave en un texto claro y coherente.\n\nMensaje:\n{user_input}"),



            #("system","Eres un experto en psicología forense que analiza mensajes escritos para detectar maltrato psicológico. Cuando el usuario te pida analizar un mensaje, identifica de manera objetiva y neutral cualquier indicio de maltrato psicológico y explica tu razonamiento. Presenta un análisis exhaustivo en español, explicando cada conclusión clave en un texto claro y coherente."),

            ("system", "Eres un experto en psicología forense que analiza mensajes escritos para detectar maltrato psicológico. Existe un historial confirmado y recurrente de maltrato psicológico hacia la persona receptora de los mensajes, por lo que debes mantener una vigilancia especial y un umbral bajo para señalar cualquier indicio, por sutil que sea. Cuando el usuario te pida analizar un mensaje, identifica de manera objetiva y neutral cualquier indicio de maltrato psicológico y explica tu razonamiento. Presenta un análisis exhaustivo en español, explicando cada conclusión clave en un texto claro y coherente."),
            ("human", "Analiza el siguiente mensaje en busca de indicios de maltrato psicológico y explica tu razonamiento.\n\nMensaje:\n{user_input}"),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    parser = StrOutputParser()

    return prompt | llm | parser


def main(argv: list[str] | None = None) -> None:
    user_input = """Querida Lucía,

Ya te di tres semanas, luego, a tu petición, tres semanas y dos días, ¿y ahora quieres uno más?

Entiendo, sin embargo, que sea más fácil avanzar por la carretera el domingo, y recuerdo esos largos trayectos. Como repites que quieres avanzar lo antes posible con el programa de vacaciones, te propongo llevar a los niños el sábado por la noche, el 5 de julio, para que tengas un día completo de viaje sin camiones al día siguiente. También es mejor para los niños. Pero no quiero cambiar toda la planificación y el cálculo de días, ya que la propuesta de mi abogada fue la última. Me devolverás esos dos días "comodín" cuando tenga un impedimento laboral. Los niños estarán igual de bien contigo que con una niñera, como ya lo he dicho varias veces.

Por otro lado, acepto tu propuesta según la cual —excepto el 5 de julio— cada uno dejará a los niños en casa del otro a las 10 de la mañana. Te dejo a ti el cuidado de poner todo esto en el calendario compartido, conforme a mi última carta del abogado y a esta conversación.

Espero tu respuesta para cerrar rápidamente el tema y te deseo un buen día.

Francisco"""

    load_dotenv()

    chain = build_chain()
    response: str = chain.invoke({"user_input": user_input})

    print(response)


if __name__ == "__main__":
    main()
