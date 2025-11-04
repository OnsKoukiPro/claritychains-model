from langchain.tools import Tool
from langchain_experimental.tools import PythonREPLTool

python_repl = PythonREPLTool()
calculator_tool = Tool(
    name="Calculator",
    func=python_repl.run,
    description="A calculator. Use this for any math calculations. DONT USE THIS TO EXECUTE PYTHON SCRIPTS!.",
)

