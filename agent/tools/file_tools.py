from langchain.tools import Tool
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
import os


def read_pdf(file_path):
    file_path = os.path.normpath(os.path.abspath(file_path.strip()))
    print(file_path)
    return PyPDFLoader(file_path).load()


def read_csv(file_path):
    file_path = os.path.normpath(os.path.abspath(file_path.strip()))
    print(file_path)
    return CSVLoader(file_path).load()


def read_docx(file_path):
    file_path = os.path.normpath(os.path.abspath(file_path.strip()))
    print(file_path)
    return UnstructuredWordDocumentLoader(file_path).load()


def read_xlsx(file_path):
    file_path = os.path.normpath(os.path.abspath(file_path.strip()))
    print(file_path)
    return UnstructuredExcelLoader(file_path).load()


file_reading_tools = [
    Tool(
        name="read_pdf",
        func=read_pdf,
        description="Reads a PDF file and returns its content. Input should be the file path.",
    ),
    Tool(
        name="read_csv",
        func=read_csv,
        description="Reads a CSV file and returns its content. Input should be the file path.",
    ),
    Tool(
        name="read_docx",
        func=read_docx,
        description="Reads a .doc or .docx file and returns its content. Input should be the file path.",
    ),
    Tool(
        name="read_xlsx",
        func=read_xlsx,
        description="Reads a .xlsx file and returns its content. Input should be the file path.",
    ),
]
