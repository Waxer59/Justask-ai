from langchain_core.prompts import ChatPromptTemplate

social_skills_prompt = ChatPromptTemplate(
    [
        ("human", """
You are a perceptive assistant. Based only on the context provided, generate a single, thoughtful question that assesses the user's social and interpersonal skills.

Context:
{context}

Instructions:
- Your question must focus on communication, empathy, teamwork, or conflict resolution.
- Use only the information in the context to guide your question.
- If the context lacks information, generate a general but insightful question.
- Write the question strictly in the same language as the context.
- Do not provide explanations, additional comments, or prefaces. Only output the question.
"""),
    ]
)

technical_skills_prompt = ChatPromptTemplate(
    [
        ("human", """
You are a technical evaluator. Based only on the context provided, generate a single, clear, and relevant technical question to assess the user's skills.

Context:
{context}

Instructions:
- Your question must target specific technical competencies, problem-solving, or technology understanding.
- Only use the information available in the context to guide your question.
- If technical details are missing, generate a general but meaningful technical question.
- Write the question strictly in the same language as the context.
- Do not provide explanations, additional comments, or prefaces. Only output the question.
"""),
    ]
)

interview_prompt = ChatPromptTemplate(
    [
        ("human", """
You are an experienced interviewer. Based only on the context provided, generate a single, realistic interview question.

Context:
{context}

Instructions:
- Your question must explore the user's background, motivation, work style, or role suitability.
- Only use the information in the context to inspire your question.
- If the context is too vague, generate a general but meaningful interview question.
- Write the question strictly in the same language as the context.
- Do not provide explanations, additional comments, or prefaces. Only output the question.
"""),
    ]
)

feedback_prompt = ChatPromptTemplate(
    [
        ("human", """
You are a constructive and supportive evaluator. Your primary task is to provide clear and actionable feedback about the user's response given in the "Important Information" section.

Important Information (User Response to Evaluate):
{question}

Instructions:
- Focus your feedback on the user's response in the "Important Information" section.
- Use the context only to enrich or clarify your evaluation if necessary, but prioritize the user's response.
- Provide detailed feedback highlighting strengths and specific areas for improvement.
- Be clear and actionable, offering concrete suggestions on how to improve.
- If the user response lacks sufficient detail to evaluate, say so explicitly.
- Write the feedback strictly in the same language as the user's response.
- Do not generate any questions or general comments. Provide only a direct feedback message on the user's response.
- Do not provide explanations, additional comments, or prefaces. Only output the feedback.
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
