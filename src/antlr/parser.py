import antlr4
from antlr.JavaLexer import JavaLexer
from antlr.JavaParser import JavaParser
from antlr.JavaParserListener import JavaParserListener
from .MyListener import KeyPrinter
import pprint

from antlr4 import RuleContext

"""
Return ast-representation from source-code
"""
class Parser:
    
    def __init__(self, ID, source_code):
        self.source_code = source_code
        self.ID = ID
        self.ast = ""


    def parse_to_ast(self):
        #print("parsing: ", self.ID)
        code_stream = antlr4.InputStream(self.source_code)
        lexer = JavaLexer(code_stream)
        token_stream = antlr4.CommonTokenStream(lexer)
        parser = JavaParser(token_stream)
        tree = parser.compilationUnit()
        
        printer = KeyPrinter()
        walker = antlr4.ParseTreeWalker()
        walker.walk(printer, tree)

        return printer.get_result()
    
