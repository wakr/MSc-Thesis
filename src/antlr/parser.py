import antlr4
from antlr.JavaLexer import JavaLexer
from antlr.JavaParser import JavaParser
from antlr.JavaParserListener import JavaParserListener
import pprint

from antlr4 import RuleContext

"""
Return ast-representation from source-code
"""
class Parser:
    
    def __init__(self, source_code):
        self.source_code = source_code
        self.ast = ""

    def ast_walker(self, p, t, indent):
        self._explore(p, t, indent)
        return self.ast

    def _explore(self, p, t, indent):
        rule_name = p.ruleNames[t.getRuleIndex()]
        self.ast = self.ast + (" " * indent) + rule_name + "\n"
        
        for j in range(t.getChildCount()):
            elem = t.getChild(j)
            if type(elem) != antlr4.tree.Tree.TerminalNodeImpl:  # the leaf
                self._explore(p, elem, indent + 2)
            else:
                self.ast = self.ast + (" " * (indent + 2)) + "${}$".format(elem.getText()) + "\n"

    def parse_to_ast(self):
        code_stream = antlr4.InputStream(self.source_code)
        lexer = JavaLexer(code_stream)
        token_stream = antlr4.CommonTokenStream(lexer)
        parser = JavaParser(token_stream)
        tree = parser.compilationUnit()
        #print("Tree " + tree.toStringTree(recog=parser))
        #return self.ast_walker(parser, tree, 0)
        return tree.toStringTree(recog=parser)
