import antlr4
from antlr.JavaLexer import JavaLexer
from antlr.JavaParser import JavaParser
from antlr.JavaParserListener import JavaParserListener
from MyListener import KeyPrinter
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

    def ast_walker(self, p, t, indent):
        self._explore(p, t, indent)
        return self.ast

    def _explore(self, p, t, indent):
        try:
            rule_name = p.ruleNames[t.getRuleIndex()]
            self.ast = self.ast + (" " * indent) + rule_name + "\n"
            
            for j in range(t.getChildCount()):
                elem = t.getChild(j)
                if type(elem) != antlr4.tree.Tree.TerminalNodeImpl:  # the leaf
                    self._explore(p, elem, indent + 2)
                else:
                    self.ast = self.ast + (" " * (indent + 2)) + "${}$".format(elem.getText()) + "\n"
        except:
            pass

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
        print(printer.get_result())
        return ""
        #return tree.toStringTree(recog=parser)


with open('example.java', 'r') as myfile:
  data = myfile.read()
  from preprocess.normalizer import normalize_for_ast
  data = normalize_for_ast(data)
  print(data)
  print(Parser(0, data).parse_to_ast())