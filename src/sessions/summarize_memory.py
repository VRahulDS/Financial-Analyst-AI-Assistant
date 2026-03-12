from langchain_core.messages import HumanMessage


def summarize_history(llm, history):

    conversation = "\n".join(
        [msg.content for msg in history.messages]
    )

    prompt = f"""
    Summarize the following conversation briefly.

    {conversation}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content