from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chat_models import ChatOpenAI

prompt_template = "What is a word to replace the following: {word}?"

# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

output_parser = CommaSeparatedListOutputParser()
template = """List all possible words as substitute for 'artificial' as comma separated."""

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=template, output_parser=output_parser, input_variables=[]),
    output_parser=output_parser)

print(f'{llm_chain.predict() = }')