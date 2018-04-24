# -*- coding: utf-8 -*-

from JavaParserListener import JavaParserListener
from JavaParser import JavaParser

class KeyPrinter(JavaParserListener):    
    
    def __init__(self):
        self.res = ""
    
    def _add_token(self, tkn):
        return self.res + tkn + "\n"
    
    def enterVariableDeclarators(self, ctx):
        self.res = self._add_token("VARDEF ")
        
    def enterAnnotation(self, ctx):
        self.res = self._add_token("ANNO    ")
        
    def enterClassDeclaration(self, ctx):
        self.res = self._add_token("CLASS{  ")
        
    def exitClassDeclaration(self, ctx):
        self.res = self._add_token("}CLASS  ")
            
    def enterImportDeclaration(self, ctx):
        self.res = self._add_token("IMPORT  ")
        
    def enterPackageDeclaration(self, ctx):
        self.res = self._add_token("PACKAGE ")
        
    def enterConstantDeclarator(self, ctx):
        self.res = self._add_token("VARDEF ")
        
    def enterClassCreatorRest(self, ctx):
        if ctx.classBody():
            self.res = self._add_token("INCLASS{")
            
    def exitClassCreatorRest(self, ctx):
        if ctx.classBody():
            self.res = self._add_token("}INCLASS")
                
    def enterCatchClause(self, ctx):
        self.res = self._add_token("CATCH{  ")
        
    def exitCatchClause(self, ctx):
        self.res = self._add_token("}CATCH  ")
            
    def enterEnumConstants(self, ctx):
        self.res = self._add_token("ENUM_CLA")
        
    def enterEnumDeclaration(self, ctx):
        self.res = self._add_token("ENUM{   ")
        
    def exitEnumDeclaration(self, ctx):
        self.res = self._add_token("}ENUM   ")
            
    def enterExplicitConstructorInvocation(self, ctx):
        self.res = self._add_token("APPLY   ")
        
    def enterVariableInitializer(self, ctx):
        if not isinstance(ctx.parentCtx, JavaParser.ArrayInitializerContext):
            self.res = self._add_token("APPLY   ")
            
    def enterCreator(self, ctx):
        if ctx.classCreatorRest():
            self.res = self._add_token("NEWCLASS")
        elif ctx.arrayCreatorRest():
            self.res = self._add_token("NEWARRAY")
    
    def enterTryStatement(self, ctx):
        self.res = self._add_token("TRY{    ")        
            
    def exitTryStatement(self, ctx):
        self.res = self._add_token("FINALLY ")   
        
    def enterNormalInterfaceDeclaration(self, ctx):
        self.res = self._add_token("INTERF{ ")        
            
    def exitNormalInterfaceDeclaration(self, ctx):
        self.res = self._add_token("}INTERF ") 
            
    def enterExplicitGenericInvocation(self, ctx):
        self.res = self._add_token("APPLY   ") 
                    
    def enterMethodDeclaration(self, ctx):
        if ctx.start.text == "void":
            self.res = self._add_token("VOID    ")
        self.res = self._add_token("METHOD{ ")
    
    def exitMethodDeclaration(self, ctx):
        self.res = self._add_token("}METHOD ")
            
    def exitSwitchLabel(self, ctx):
        self.res = self._add_token("CASE    ")

    def enterConstructorDeclaration(self, ctx):
        self.res = self._add_token("CONSTR{ ")
         
    def exitConstructorDeclaration(self, ctx):
        self.res = self._add_token("}CONSTR ")
            
    def enterResource(self, ctx):
        self.res = self._add_token("TRY_RES ")
            
    def enterArrayInitializer(self, ctx):
        self.res = self._add_token("ARRINIT{")
    
    def exitArrayInitializer(self, ctx):
        self.res = self._add_token("ARRINIT}")
            
    def enterTypeArgument(self, ctx):
        if isinstance(ctx.parentCtx.parentCtx, JavaParser.CreatedNameContext):
            pass
        else:
            self.res = self._add_token("GENERIC ")
    
    def enterDoWhileStmt(self, ctx):
            self.res = self._add_token("LOOP{ ")
    
    def exitDoWhileStmt(self, ctx):
            self.res = self._add_token("}LOOP ")
    
    def enterBreak(self, ctx):
                self.res = self._add_token("BREAK   ")
    
    def enterIfStmt(self, ctx):
                self.res = self._add_token("IF{     ")
    
    def exitIfStmt(self, ctx):
                self.res = self._add_token("}IF     ")
    


        
    def get_result(self):
        return self.res