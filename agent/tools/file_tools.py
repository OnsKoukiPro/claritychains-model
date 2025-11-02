from langchain.tools import Tool
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
import os
import pandas as pd


def read_pdf(file_path):
    """Read PDF file and return text content as string"""
    try:
        file_path = os.path.normpath(os.path.abspath(file_path.strip()))
        print(f"Reading PDF: {file_path}")
        docs = PyPDFLoader(file_path).load()
        # Extract text content from documents and join into a single string
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error reading PDF file {file_path}: {str(e)}"


def read_csv(file_path):
    """Read CSV file and return content as string"""
    try:
        file_path = os.path.normpath(os.path.abspath(file_path.strip()))
        print(f"Reading CSV: {file_path}")
        # Use pandas for more reliable CSV reading
        df = pd.read_csv(file_path)
        return df.to_string()
    except Exception as e:
        return f"Error reading CSV file {file_path}: {str(e)}"


def read_docx(file_path):
    """Read DOCX file and return text content as string"""
    try:
        file_path = os.path.normpath(os.path.abspath(file_path.strip()))
        print(f"Reading DOCX: {file_path}")
        docs = UnstructuredWordDocumentLoader(file_path).load()
        # Extract text content from documents and join into a single string
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error reading DOCX file {file_path}: {str(e)}"


def read_xlsx(file_path):
    """Read Excel file and return content as string"""
    try:
        file_path = os.path.normpath(os.path.abspath(file_path.strip()))
        print(f"Reading Excel: {file_path}")
        # Use pandas for Excel reading
        df = pd.read_excel(file_path)
        return df.to_string()
    except Exception as e:
        return f"Error reading Excel file {file_path}: {str(e)}"


def read_txt(file_path):
    """Read text file and return content as string"""
    try:
        file_path = os.path.normpath(os.path.abspath(file_path.strip()))
        print(f"Reading text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading text file {file_path}: {str(e)}"


file_reading_tools = [
    Tool(
        name="read_pdf",
        func=read_pdf,
        description="Reads a PDF file and returns its text content. Input should be the file path.",
    ),
    Tool(
        name="read_csv",
        func=read_csv,
        description="Reads a CSV file and returns its content. Input should be the file path.",
    ),
    Tool(
        name="read_docx",
        func=read_docx,
        description="Reads a .doc or .docx file and returns its text content. Input should be the file path.",
    ),
    Tool(
        name="read_xlsx",
        func=read_xlsx,
        description="Reads a .xlsx file and returns its content. Input should be the file path.",
    ),
    Tool(
        name="read_txt",
        func=read_txt,
        description="Reads a text file and returns its content. Input should be the file path.",
    ),
]