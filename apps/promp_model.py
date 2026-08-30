from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("Say {foo}")
prompt.format(foo = "bar")
prompt = PromptTemplate(template = "Say {foo}")
