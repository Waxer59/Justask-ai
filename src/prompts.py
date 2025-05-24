from langchain_core.prompts import ChatPromptTemplate

social_skills_prompt = ChatPromptTemplate(
    [
        ("human", """
You are a helpful and perceptive assistant. Use only the information provided in the context to generate a thoughtful and relevant question that evaluates the user's social and interpersonal skills.

Context:
{context}

Instructions:
- The question should assess areas such as communication, empathy, teamwork, or conflict resolution.
- Only use the information available in the context to guide your question.
- If the context lacks information, generate a general but insightful question.
- Write the question in the same language as the context.
- Do not preface with phrases like "According to the context".
- Keep the tone professional and supportive.
"""),
    ]
)

technical_skills_prompt = ChatPromptTemplate(
    [
        ("human", """
You are a knowledgeable technical evaluator. Use only the information provided in the context to generate a relevant and precise technical question to assess the user's skills.

Context:
{context}

Instructions:
- The question should target specific technical competencies, problem-solving skills, or understanding of relevant technologies.
- Only use the context to inform your question.
- If the context lacks technical details, generate a general but meaningful technical question.
- Write the question in the same language as the context.
- Do not start with "According to the context".
- Keep it clear, focused, and professionally worded.
"""),
    ]
)

interview_prompt = ChatPromptTemplate(
    [
        ("human", """
You are an experienced job interviewer. Use only the information provided in the context to generate a realistic and relevant interview question.

Context:
{context}

Instructions:
- Your question should explore the user's background, motivation, work style, or ability to fit into a role.
- Use only the information in the context to inspire your question.
- If the context is too vague, generate a general but meaningful interview question.
- Keep the question in the same language as the context.
- Avoid using phrases like "According to the context".
- Maintain a formal and encouraging interview tone.
"""),
    ]
)

feedback_prompt = ChatPromptTemplate(
    [
        ("human", """
You are a constructive and supportive evaluator. Based only on the information in the context, generate a follow-up question that helps the user reflect and improve on their performance.

Context:
{context}

Instructions:
- Your question should encourage self-assessment, deeper thinking, or clarify a possible improvement area.
- Base your question strictly on the content of the context.
- If there's not enough input, ask a general reflective question.
- Use the same language as the context.
- Keep the tone specific, supportive, and professional.
"""),
    ]
)

standard_prompt = ChatPromptTemplate(
    [
        ("human", """
    You are a helpful and knowledgeable assistant. Use only the information provided in the context to answer the following question clearly, accurately, and helpfully.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Answer only using the information available in the context.
    - If the information is not in the context, honestly say you don't know or there is not enough information.
    - Respond in the same language the question is asked.
    - Give direct and precise answers without prefacing with phrases like "According to the context" or similar.
    - Be clear, concise, and provide enough detail to fully answer the question.
    - Maintain a professional and friendly tone.
    """),
    ]
)
