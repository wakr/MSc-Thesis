import antlr4
from antlr.JavaLexer import JavaLexer
from antlr.JavaParser import JavaParser
from antlr.JavaParserListener import JavaParserListener
import pprint

from antlr4 import RuleContext

#%%

def ASTWalker(p, t, indent):
    explore(p, t, indent)
    
    
def explore(p, t, indent):
    ruleName = p.ruleNames[t.getRuleIndex()]
    for i in range(indent):
        print(" ", end="")
    print(ruleName)
    for j in range(t.getChildCount()):
        elem = t.getChild(j)
        if type(elem) != antlr4.tree.Tree.TerminalNodeImpl: # the leaf
            explore(p, elem, indent + 1)



#%%


code = open('./example.java', 'r').read()

codeStream = antlr4.InputStream(code)
lexer = JavaLexer(codeStream)
tokensStream = antlr4.CommonTokenStream(lexer)
parser = JavaParser(tokensStream)

tree = parser.compilationUnit()
print("Tree " + tree.toStringTree(recog=parser))

#ASTWalker(parser, tree, 0)

#printer = JavaParserListener()
#walker = ParseTreeWalker()
#walker.walk(printer, tree)
#%%