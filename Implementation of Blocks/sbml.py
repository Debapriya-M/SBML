#Debapriya Mukherjee, SBUID : 112683344

import ply.lex as lex
import ply.yacc as yacc
import sys

dict_variable = {}

class SemanticError(Exception):
    pass

class Node:
    def __init__(self):
        # print("init node")
        self.value = 0

    def evaluate(self):
        return self.value

    def execute(self):
        return self.value

class NumberNode(Node):
    def __init__(self, v):
        if('.' in v or 'e' in v or 'E' in v):
            self.value = float(v)
        else:
            self.value = int(v)

    def evaluate(self):
        return self.value

    def execute(self):
        return self.value

class StringNode(Node):
    def __init__(self, v):
        
        self.value = str(v)

    def evaluate(self):
        return self.value

    def execute(self):
#        return '\'' + self.value + '\'' #Implementation for HW3
        return self.value

class VariableNode(Node):
    def __init__(self, v):
        self.v = v

    def evaluate(self):
        return dict_variable[self.v]  #NONE get

    def execute(self):
        return self.evaluate()

class AssignmentNode(Node):
    def __init__(self, v1, v2):
            self.v1 = v1
            self.v2 = v2

    def evaluate(self):
        #for checking if the assignment node is a part of a list. If so, it will add the element to the list.
        if (isinstance(self.v1, IndexListNode)):
            tempvar = self.v1.v1
            index = self.v1.v2.evaluate()
            templist = tempvar.evaluate()
            templist[index] = self.v2.evaluate()
            dict_variable[tempvar] = templist
        elif (isinstance(self.v1, VariableNode)):
            dict_variable[self.v1.v] = self.v2.evaluate()
        else:
            raise SemanticError

    def execute(self):
        return self.evaluate()

##Implementing the block with braces
class BlockNode(Node):
    def __init__(self,sl):
        self.statementList = sl

    def evaluate(self):
        if (self.statementList == None):
            None
        else:
            for statement in self.statementList:
                statement.evaluate()

    def execute(self):
        return self.evaluate()

#Implementing the if block only
class IfNode(Node):
    def __init__(self, condition, block):
        self.condition = condition
        self.block = block

    def evaluate(self):
        if (isinstance(self.condition.evaluate(), bool) and isinstance(self.block, BlockNode)):
            if (self.condition.evaluate()):
                self.block.evaluate()
        else:
            raise SemanticError

    def execute(self):
        return self.evaluate()

#Implementing the else block
class ElseNode(Node):
    def __init__(self, block1, block2):
        self.block1 = block1
        self.block2 = block2

    def evaluate(self):
        if (isinstance(self.block1, IfNode) and isinstance(self.block2, BlockNode)):
            if (self.block1.condition.evaluate()):
                self.block1.evaluate()
            else:
                self.block2.evaluate()
        else:
            raise SemanticError

    def execute(self):
        return self.evaluate()

#Implementing the while loop
class WhileNode(Node):
    def __init__(self, cond, block):
        self.cond = cond
        self.block = block

    def evaluate(self):
        if (isinstance(self.block, BlockNode) and isinstance(self.cond.evaluate(), bool)):
            while(self.cond.evaluate()):
                self.block.evaluate()
        else:
            raise SemanticError

    def execute(self):
        return self.evaluate()

class PrintNode(Node):
    def __init__(self, v):
        self.v = v

    def evaluate(self):
        print(self.v.evaluate())

    def execute(self):
        return self.evaluate()

class BopNode(Node):
    def __init__(self, op, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.op = op
        self.value = 0
        
    def typecheck(self):
        v1 = self.v1.evaluate()
        v2 = self.v2.evaluate()
        # print("v1: ", v1)
        # print("v2: ", v2)
        if type(v1) != int and type(v1) != float:
            raise SemanticError()
        elif type(v2) != int and type(v2) != float:
            raise SemanticError()
        else:
            pass

    def evaluate(self):
        # print("Inside BopNode evaluate")
        v1 = self.v1.evaluate()
        v2 = self.v2.evaluate()
        if self.op == '+':
            # print("Inside + operator")
            if type(v1) == int or type(v1) == float:
                if type(v2) != int and type(v2) != float:
                    raise SemanticError()
                else:
                    pass
            elif type(v1) != str or type(v1) != list:
                if type(v1) != type(v2):
                    raise SemanticError()
            self.value = v1 + v2
        elif self.op == '-':
            # print("Inside - operator")
            self.typecheck()
            self.value = v1 - v2
        elif self.op == '*':
            # print("Inside * operator")
            self.typecheck()
            self.value = v1 * v2
        elif self.op == '/':
            # print("Inside / operator")
            self.typecheck()
            if v2 == 0:
                raise SemanticError()
            self.value = v1 / v2
        elif self.op == '**':
            # print("Inside ** operator")
            self.typecheck()
            self.value = v1 ** v2
        elif self.op == 'mod':
            # print("Inside mod operator")
            self.typecheck()
            # print()
            self.value = v1 % v2
        elif self.op == 'div':
            # print("Inside div operator")
            self.typecheck()
            self.value = v1 // v2    
            # print(self.value)
        return self.value

    def execute(self):
        if (isinstance(self.v1.evaluate(), str)):
            return '\''+self.evaluate()+'\''
        return self.evaluate()

class ComparisonNode(Node):
    def __init__(self, comparator, v1, v2):
        super().__init__()
        self.comparator = comparator
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        v1 = self.v1.evaluate()
        v2 = self.v2.evaluate()
        # self.typeChecking()
        if type(v1) == int or type(v1) == float:
            if type(v2) != int and type(v2) != float:
                raise SemanticError()
            else:
                pass
        if type(v1) == str or type(v2) == str:
            if type(v2) != type(v1):
                raise SemanticError()
            else:
                pass
        if self.comparator == '<':
            self.value = (v1 < v2)
            return self.value
        elif self.comparator == '>':
            self.value = (v1 > v2)
            return self.value
        elif self.comparator == '<=':
            self.value = (v1 <= v2)
            return self.value
        elif self.comparator == '>=':
            self.value = (v1 >= v2)
            return self.value
        elif self.comparator == '==':
            self.value = (v1 == v2)
            return self.value
        elif self.comparator == '<>':
            self.value = (v1 < v2 or v1 > v2)
            return self.value
        # print("+++++",self.value)
        else:
            raise SemanticError

    def execute(self):
        return self.evaluate()

class BooleanOpNode(Node):
    def __init__(self, comparator, v1, v2):
        super().__init__(False)
        self.comparator = comparator
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        # print("Inside the boolean operator node")
        v1 = self.v1.evaluate()
        v2 = self.v2.evaluate()
        if not (isinstance(v1, bool) and isinstance(v2, bool)):
            raise ValueError
        try:
            if self.comparator == 'andalso':
                self.value = (v1 and v2)
            elif self.comparator == 'orelse':
                self.value = (v1 or v2)
            return self.value
        except Exception:
            raise ValueError("SEMANTIC ERROR")

    def execute(self):
        self.evaluate()
        # print("compare execute")
        return self.value

class NotOp(Node):
    def __init__(self, op, v1):
        self.v1 = v1
        self.op = op

    def evaluate(self):
        if (isinstance(self.v1.evaluate(), bool)):
            return not self.v1.evaluate()
        else:
            raise SemanticError

    def execute(self):
        return self.evaluate()

class IsMemberNode(Node):
    def __init__(self, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.value = 0

    def evaluate(self):
        v1 = self.v1.evaluate()
        v2 = self.v2.evaluate()
        # print("Inside IsMemberNode evaluate")
        if type(v2) != type("") and type(v2) != type([]):
            raise SemanticError()
        self.value = v1 in v2
        return self.value

    def execute(self):
        return self.evaluate()

class ListNode1(Node):
    def __init__(self, v1):
        self.v1 = v1

    def evaluate(self):
        if (self.v1 == None):
            return []
        else:
            return [self.v1.evaluate()]

    def execute(self):
        return self.evaluate()

class ListNode2(Node):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        return self.v1.evaluate() + [self.v2.evaluate()]

    def execute(self):
        return self.evaluate()

class TupleNode(Node):
    def __init__(self, l):
        super().__init__()
        self.value = tuple(l)

    def get_element(self, index):
        if index <= 0:
            raise Exception("SEMANTIC ERROR")
        return self.value[index - 1]

    def evaluate(self):
        return tuple([x.evaluate() for x in self.value])

    def execute(self):
        return tuple([x.execute() for x in self.value])

class NegativeNode(Node):
    def __init__(self, v1):
        self.v1 = v1

    def evaluate(self):
        if (isinstance(self.v1.evaluate(), float) or isinstance(self.v1.evaluate(), int)):
            return -1 * self.v1.evaluate()
        else:
            raise SemanticError

    def execute(self):
        return self.evaluate()


class ConcatNode(Node):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        if (isinstance(self.v2.evaluate(), list)):
            return [self.v1.evaluate()] + self.v2.evaluate()
        else:
            raise SemanticError

    def execute(self):
        return self.evaluate()

#This is for finding the index of the list.
class IndexListNode(Node):
    def __init__(self, v1, v2):
            self.v1 = v1
            self.v2 = v2

    def evaluate(self):
        if ((isinstance(self.v1.evaluate(), list) or isinstance(self.v1.evaluate(), str)) and isinstance(self.v2.evaluate(), int)):
            return self.v1.evaluate()[self.v2.evaluate()]
        else:
            raise SemanticError

    def execute(self):
        return self.evaluate()

#This is for finding the index of the tuple.
class IndexTupleNode(Node):
    def __init__(self, v, index):
        super().__init__()
        self.v = v
        self.index = index
        self.value = 0

    def evaluate(self):
        # print("Inside TupleNode evaluate")
        index = self.index.evaluate()
        # print(index)
        v = self.v.evaluate()
        # print(v)
        if index <= 0:
            raise SemanticError
        self.value = v[index - 1]
        # print(self.value)
        return self.value

    def execute(self):
        return self.evaluate()

class BooleanNode(Node):
    def __init__(self, v):
        if (v == 'True'):
            self.v = True
        elif (v == 'False'):
            self.v = False
        else:
            raise SemanticError

    def evaluate(self):
        return self.v

    def execute(self):
        return self.v

reserved = {
    'if'    : 'IF',
    'else'  : 'ELSE',
    'while' : 'WHILE',
    'print' : 'PRINT',
    'mod'   : 'MOD',
    'div'   : 'DIV',
    'in'    : 'IN',
    'not'   : 'NOT',
    'andalso' : 'ANDALSO',
    'orelse'  : 'ORELSE',
}

tokens = [
    'LPAREN', 'RPAREN', 'NUMBER',
    'PLUS','MINUS','TIMES','DIVIDE','BOOLEAN',
    'LBRACKET', 'RBRACKET', 'STRING', 'COMMA',
    'TUPLE_INDEX', 'POWER', 'CONCAT',
    'LESS', 'LESSEQUAL', 'EQUAL', 'NOTEQUAL', 'GREATER', 'GREATEREQUAL',
    'SEMICOLON', 'VARIABLE', 'ASSIGN',
    'LCURLY', 'RCURLY',
    ]

tokens = tokens + list(reserved.values())

# Tokens
t_ASSIGN  = r'='
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = r','
t_TUPLE_INDEX = r'\#'
t_POWER = r'\*\*'
t_CONCAT = r'::'
t_LESS = r'<'
t_LESSEQUAL = r'<='
t_EQUAL = r'=='
t_NOTEQUAL = r'<>'
t_GREATER = r'>'
t_GREATEREQUAL = r'>='
t_SEMICOLON = r';'
t_LCURLY = r'\{'
t_RCURLY = r'\}'

def t_NUMBER(t):
    r'-?\d*(\d\.|\.\d|\d)\d*[Ee](-|\+)?\d+|\d*(\d\.|\.\d)\d*|\d+'
    try:
        t.value = NumberNode(t.value)
    except ValueError:
        raise SyntaxError
        t.value = 0
    return t

def t_STRING(t):
    r'(\"([^\\\n]|(\\.))*?\") | (\'([^\\\n]|(\\.))*?\')'
    try:
        t.value = StringNode(t.value[1:-1])
    except:
        raise SyntaxError
    return t

def t_BOOLEAN(t):
    'True|False'
    t.value = BooleanNode(t.value)
    return t

def t_VARIABLE(t):
    r'[A-Za-z][A-Za-z0-9_]*'
    try:
        if t.value in reserved:
            #this will check the variable name with the reserved list of tokens.
            t.type = reserved[t.value]
        else:
            t.value = VariableNode(t.value)
    except:
        raise SyntaxError
    return t

# Ignored characters
t_ignore = " \t"

def t_error(t):
    raise SyntaxError
#    print("Syntax error at '%s'" % t.value)

# Build the lexer

lex.lex(debug=0)

# Parsing rules
precedence = (
    ('left', 'ORELSE'),
    ('left', 'ANDALSO'),
    ('left', 'NOT'),
    ('left', 'GREATER', 'GREATEREQUAL', 'LESS', 'LESSEQUAL', 'EQUAL', 'NOTEQUAL'),
    ('right', 'CONCAT'),
    ('left', 'IN'),
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE', 'MOD', 'DIV'),
    ('left', 'UMINUS'),
    ('right', 'POWER'),
    ('left', 'RBRACKET', 'LBRACKET'),
    ('left', 'TUPLE_INDEX'),
    ('left', 'LPAREN', 'RPAREN')
    )

def p_block(t):
    '''block : LCURLY statement_list RCURLY'''
    t[0] = BlockNode(t[2])

def p_block2(t):
    '''block : LCURLY RCURLY'''
    t[0] = BlockNode(None)

def p_statement_list(t):
    '''statement_list : statement_list statement'''
    t[0] = t[1] + [t[2]]

def p_statement_list_val(t):
    '''statement_list : statement'''
    t[0] = [t[1]]

def p_statement(t):
    '''statement : block'''
    t[0] = t[1]

def p_if_statement(t):
    '''if_statement : IF LPAREN expression RPAREN block'''
    t[0] = IfNode(t[3], t[5])

def p_statement_if(t):
    '''statement : if_statement'''
    t[0] = t[1]

def p_if_else_statement(t):
    '''else_statement : if_statement ELSE block'''
    t[0] = ElseNode(t[1], t[3])

def p_statement_else(t):
    '''statement : else_statement'''
    t[0] = t[1]

def p_while_statement(t):
    '''while_statement : WHILE LPAREN expression RPAREN block'''
    t[0] = WhileNode(t[3], t[5])

def p_statement_while(t):
    '''statement : while_statement'''
    t[0] = t[1]

def p_print_statement(t) :
    '''statement : PRINT LPAREN expression RPAREN SEMICOLON'''
    t[0] = PrintNode(t[3])

def p_statement_expression_semicolon(t):
    '''statement : expression SEMICOLON'''
    t[0] = t[1]

def p_assignment(t):
    '''statement : expression ASSIGN expression SEMICOLON'''
    t[0] = AssignmentNode(t[1], t[3])

def p_expression_group(t):
    '''expression : LPAREN expression RPAREN'''
    t[0] = t[2]

def p_expression_binop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression DIV expression
                  | expression POWER expression
                  | expression MOD expression'''
    t[0] = BopNode(t[2], t[1], t[3])

def p_expression_uminus(t):
    '''expression : MINUS expression %prec UMINUS'''
    t[0] = NegativeNode(t[2])

def p_expression_operator(t):
    '''expression : expression GREATER expression
                  | expression GREATEREQUAL expression
                  | expression LESS expression
                  | expression LESSEQUAL expression
                  | expression EQUAL expression
                  | expression NOTEQUAL expression'''
    t[0] = ComparisonNode(t[2], t[1], t[3])

def p_expression_booleanop(t):
    '''expression : expression ANDALSO expression
                  | expression ORELSE expression'''
    t[0] = BooleanOpNode(t[2], t[1], t[3])

def p_concat(t):
    '''expression : expression CONCAT expression'''
    t[0] = ConcatNode(t[1], t[3])

def p_indexList(t):
    '''expression : expression LBRACKET expression RBRACKET'''
    t[0] = IndexListNode(t[1], t[3])

def p_expression_notop(t):
    '''expression : NOT expression'''
    t[0] = NotOp(t[1], t[2])

def p_expression_in(t):
    '''expression : expression IN expression'''
    t[0] = IsMemberNode(t[2], t[1], t[3])

def p_expression_factor(t):
    '''expression : factor'''
    t[0] = t[1]

def p_factor_number(t):
    '''factor : NUMBER
              | STRING
              | tuple'''
    t[0] = t[1]

def p_expression_assign(t):
    '''expression : VARIABLE'''
    t[0] = t[1]

def p_expression_boolean(t):
    '''expression : BOOLEAN'''
    t[0] = t[1]

def p_noneList(t):
    '''expression : LBRACKET RBRACKET'''
    t[0] = ListNode1(None)

def p_expression_tuple(t):
    '''tuple : LPAREN tupleelements RPAREN
             | LPAREN RPAREN'''
    # print("Inside p_expression_tuple")
    if len(t) > 3:
        t[0] = TupleNode(t[2])
    else:
        t[0] = TupleNode([])

def p_expression_elements_tuple2(t):
    '''tupleelements : '''
    t[0] = []

def p_expression_elements_tuple(t):
    '''tupleelements : expression COMMA tupleelements
                | expression'''
    # print("Inside p_expression_elements_tuple")
    if len(t) > 2:
        # t[1].append(t[3])
        t[0] = [t[1]] + t[3]
    else:
        t[0] = [t[1]]

def p_expression_tuple_index(t):
    '''expression : TUPLE_INDEX expression LPAREN expression RPAREN
                  | TUPLE_INDEX expression expression '''
    # print("p_expression_tuple_index")
    index = t[2]
    if len(t) > 4:
        t[0] = IndexTupleNode(t[4],index)
    else:
        t[0] = IndexTupleNode(t[3],index)

def p_expression_lbracket_list_rbracket(t):
    '''expression : LBRACKET list RBRACKET'''
    t[0]= t[2]

def p_expression_list2(t):
    '''list : list COMMA expression'''
    t[0] = ListNode2(t[1], t[3])

def p_list(t):
    '''list : expression'''
    t[0] = ListNode1(t[1])

def p_error(t):
    raise SyntaxError

yacc.yacc()

def main():
    #checking the arguments given while running the python file is correct
    if (len(sys.argv) != 2):
        sys.exit("Invalid arguments! Please check the arguments.")
        
    #Reading the file as a complete string instead of reading it line by line, by replacing new lines with blanks.
    with open(sys.argv[1], 'r') as myfile:
        data = myfile.read().replace('\n', '')
    # print(data)
    try:
        root = yacc.parse(data)
        #calling the evaluate methods.
        root.evaluate()
    except SyntaxError as err:
        #raising syntax errors after checking
        print("SYNTAX ERROR")
    except Exception as err:
         #raising semantic errors after checking
        print("SEMANTIC ERROR")


if __name__ == "__main__":
    main()


# Citations for Reference:
# CSE 307 PLY Lecture Videos and Examples provided by Christopher Kane
# https://www.dabeaz.com/ply/ply.html#ply_nn24
# https://www.dabeaz.com/ply/example.html