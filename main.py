from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

prompt_template = "What is a word to replace the following: {word}?"

# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

output_parser = CommaSeparatedListOutputParser()
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

op = conversation.predict(input="List all possible words as substitute for 'artificial' as comma separated.")
print(op)
op_of_next4 = conversation.predict(input="And the next 4?")
print(op_of_next4)