from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI

prompt_template = "What is a word to replace the following: {word}?"

# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

print(f'{llm_chain("artificial")=}\n\n')
input_list = [
    {"word": "artificial"},
    {"word": "intelligence"},
    {"word": "robot"}
]

print(f'{llm_chain.apply(input_list) = }\n\n')

print(f'{llm_chain.generate(input_list) = }\n\n')

print(f'{llm_chain.predict(word="robot") = }')
