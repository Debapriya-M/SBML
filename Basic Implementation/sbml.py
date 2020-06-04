#Debapriya Mukherjee, SBU ID: 112683344

import ply.lex as lex
import sys
import ply.yacc as yacc
import warnings as warnings

warnings.filterwarnings("ignore")
class SemanticError(Exception):
    pass

class Node:
    def __init__(self):
        # print("Inside __init__ block")
        self.value = 0

    def evaluate(self):
        return self.value

    # def execute(self):
    #     return self.value

class BopNode(Node):
    def __init__(self, op, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.op = op
        self.value = 0
        
    def typecheck(self):
        v1 = self.v1
        v2 = self.v2
        if type(v1) != int and type(v1) != float:
            raise SemanticError()
        elif type(v2) != int and type(v2) != float:
            raise SemanticError()
        else:
            pass

    def evaluate(self):
        # print("Inside BopNode evaluate")
        v1 = self.v1
        v2 = self.v2
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
            self.value = v1 % v2
        elif self.op == 'div':
            # print("Inside div operator")
            self.typecheck()
            self.value = v1 // v2    
            # print(self.value)
        return self.value


class ExpNode(Node):
    def __init__(self, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
       
    def evaluate(self):
        # print("Inside TupleNode evaluate")
        v1 = self.v1
        v2 = self.v2
        self.value = v1 * (10**v2)
        return self.value

class BooleanNode(Node):
    def __init__(self, s):
        super().__init__()
        # print("Within Boolean --->")
        self.s = s
        # print(s)

    def evaluate(self):
        s = self.s
        # print("Inside Boolean evaluate")
        if s == 'True' or s == True:
            self.value = True
        else:
            self.value = False
        return self.value

class ComparisonNode(Node):
    def __init__(self, comparator, v1, v2):
        super().__init__()
        self.comparator = comparator
        self.v1 = v1
        self.v2 = v2

    def typeChecking(self):
        v1 = self.v1
        v2 = self.v2
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

    def evaluate(self):
        v1 = self.v1
        v2 = self.v2
        self.typeChecking()
        if self.comparator == '<':
            self.value = (v1 < v2)
        elif self.comparator == '>':
            self.value = (v1 > v2)
        elif self.comparator == '<=':
            self.value = (v1 <= v2)
        elif self.comparator == '>=':
            self.value = (v1 >= v2)
        elif self.comparator == '==':
            self.value = (v1 == v2)
        elif self.comparator == '<>':
            self.value = (v1 < v2 or v1 > v2)
        return self.value

class StringNode(Node):
    def __init__(self, s):
        super().__init__()
        self.value = str(s)

    def evaluate(self):
        return self.value


class IsMemberNode(Node):
    def __init__(self, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.value = 0

    def evaluate(self):
        v1 = self.v1
        v2 = self.v2
        # print("Inside IsMemberNode evaluate")
        if type(v2) != type("") and type(v2) != type([]):
            raise SemanticError()
        self.value = v1 in v2
        return self.value

class ConcatNode(Node):
    def __init__(self, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.value = 0

    def evaluate(self):
        # print("Inside ConcatNode evaluate")
        v1 = self.v1
        v2 = self.v2
        if type(v2) != type([]):
            raise SemanticError()
        v2.insert(0, v1)
        self.value = v2
        return self.value

class TupleNode(Node):
    def __init__(self, v, index):
        super().__init__()
        self.v = v
        self.index = index
        self.value = 0

    def evaluate(self):
        # print("Inside TupleNode evaluate")
        index = self.index
        v = self.v
        if index <= 0:
            raise Exception("SEMANTIC ERROR")
        self.value = v[index - 1]
        return self.value

class ListNode(Node):
    def __init__(self, v, index):
        super().__init__()
        self.v = v
        self.index = index
        self.value = 0

    def evaluate(self):
        # print("Inside ListNode evaluate")
        index = self.index
        v = self.v
        if type(v) == str or type(v) == list:
            pass
        else:
            raise SemanticError()
        self.value = v[index]
        return self.value

tokens = (
    'NUMBER', 
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'POWER', 'MOD', 'DIV', 'EXPONENT',
    'LPAREN', 'RPAREN',
    'LESSERERTHAN', 'GREATERTHAN', 'LESSERERTHANEQUAL', 'GREATERTHANEQUAL', 'EQUALEQUAL', 'NOTEQUAL',
    'BOOLEAN', 'STRING', 'NOT', 'IN', 'CONS',
    'LBRACKET', 'RBRACKET', 'COMMA',
    'TUPLE_INDEX', 'ORELSE', 'ANDALSO',
    )

#TOKENS
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
# t_SEMICOLON = r'\;'
t_COMMA = r','
t_POWER = r'\*\*'
t_MOD = r'mod'
t_DIV = r'div'
t_EQUALEQUAL = r'=='
t_LESSERERTHAN = r'<'
t_GREATERTHAN = r'>'
t_LESSERERTHANEQUAL = r'<='
t_GREATERTHANEQUAL = r'>='
t_NOTEQUAL = r'<>'
t_NOT = r'not'
t_IN = r'in'
t_CONS = r'::'
t_ANDALSO = r'andalso'
t_ORELSE = r'orelse'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_TUPLE_INDEX = r'\#'
t_EXPONENT = 'e'


def t_NUMBER(t):
    r'(\d+(?:\.\d+)?)'
    # print("Inside t_number:" + t.value)
    try:
        if '.' in t.value:
            t.value = float(t.value)
        else:
            t.value = int(t.value)
    except ValueError:
        raise SyntaxError("SyntaxError - number should be valid")  
        t.value = 0  
    return t

def t_BOOLEAN(t):
    r'(True)|(False)'
    # print("Inside t_BOOLEAN")
    t.value = BooleanNode(t.value).evaluate()
    return t

def t_STRING(t):
    r'(\"([^\\\n]|(\\.))*?\") | (\'([^\\\n]|(\\.))*?\')'
    # print("Inside t_string")
    try:
        t.value = StringNode(t.value[1:-1]).evaluate()
    except ValueError:
        raise SyntaxError("SYNTAX ERROR - String should be valid")
    return t

t_ignore = " \t"

def t_error(t):
    # print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
    # raise SyntaxError()

lex.lex()

precedence = (
    ('left', 'ORELSE'),
    ('left', 'ANDALSO'),
    ('left', 'NOT'),
    ('nonassoc', 'LESSERERTHAN', 'GREATERTHAN', 'LESSERERTHANEQUAL', 'GREATERTHANEQUAL', 'EQUALEQUAL', 'NOTEQUAL'),
    ('right', 'CONS'),
    ('left', 'IN'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE', 'MOD', 'DIV'),
    ('right', 'UMINUS'),
    ('right', 'POWER', 'EXPONENT'),
    ('left', 'LBRACKET', 'RBRACKET'),
    ('left', 'TUPLE_INDEX'),
    ('nonassoc', 'LPAREN', 'RPAREN'),
    )

def p_expression_group(t):
    '''expression : LPAREN expression RPAREN'''
    # print("Inside p_expression_group")
    t[0] = t[2]

def p_factor(t):
    '''factor : NUMBER
              | BOOLEAN
              | STRING
              | tuple
              | list'''
    # print("Inside p_factor: ", t[1])
    t[0] = t[1]

def p_expression_not(t):
    '''expression : NOT expression'''
    if type(True) != type(t[2]):
        raise SemanticError()
    t[0] = not t[2]

def p_expression_factor(t):
    '''expression : factor'''
    # print("Inside p_expression_factor")
    t[0] = t[1]

def p_expression_binop(t):
    '''expression : expression PLUS expression
                      | expression MINUS expression
                      | expression TIMES expression
                      | expression DIVIDE expression
                      | expression POWER expression
                      | expression MOD expression
                      | expression DIV expression'''
    # print("p_expression_binop")
    t[0] = BopNode(t[2], t[1], t[3]).evaluate()

def p_expression_exponentiation(t):
    """expression : expression EXPONENT expression"""
    t[0] = ExpNode(t[1],t[3]).evaluate()


def p_expression_comparison(t):
    """expression : expression LESSERERTHAN expression
                      | expression GREATERTHAN expression
                      | expression LESSERERTHANEQUAL expression
                      | expression GREATERTHANEQUAL expression
                      | expression EQUALEQUAL expression
                      | expression NOTEQUAL expression"""
    # print("Inside p_expression_comparison")
    t[0] = ComparisonNode(t[2], t[1], t[3]).evaluate()

def p_expression_conjunction(t):
    '''expression : expression ANDALSO expression'''
    if type(t[1]) != bool or type(t[3]) != bool:
        raise SemanticError()
    t[0] = t[1] and t[3]

def p_expression_disjunction(t):
    '''expression : expression ORELSE expression'''
    if type(t[1]) != bool or type(t[3]) != bool:
        raise SemanticError()
    t[0] = t[1] or t[3]


def p_expression_in(t):
    '''expression : expression IN expression'''
    # print("Inside p_expression_in")
    t[0] = IsMemberNode(t[1],t[3]).evaluate()

def p_expression_concatenation(t):
    '''expression : expression CONS expression'''
    # print("Inside p_expression_concatenation")
    t[0] = ConcatNode(t[1],t[3]).evaluate()

def p_expression_elements_lists(t):
    '''listelements : expression COMMA listelements
                    | expression'''
    # print("Inside p_expression_elements_lists")
    if len(t) > 2:
        # t[1].append(t[3])
        t[0] = [t[1]] + t[3]
    else:
        t[0] = [t[1]]

def p_expression_elements_lists2(t):
    '''listelements : '''
    t[0] = []

def p_expression_list(t):
    '''list : LBRACKET listelements RBRACKET
            | LBRACKET RBRACKET'''
    # print("Inside p_expression_list")
    if len(t) > 3:
        t[0] = list(t[2])
    else:
        t[0] = list()

def p_expression_list_index(t):
    '''expression : expression LBRACKET expression RBRACKET'''
    # print("Inside p_expression_list_index")
    index = t[3]
    t[0] = ListNode(t[1],index).evaluate()


def p_expression_tuple(t):
    '''tuple : LPAREN tupleelements RPAREN
             | LPAREN RPAREN'''
    # print("Inside p_expression_tuple")
    if len(t) > 3:
        t[0] = tuple(t[2])
    else:
        t[0] = tuple([])

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
        t[0] = TupleNode(t[4],index).evaluate()
    else:
        t[0] = TupleNode(t[3],index).evaluate()

def p_expression_uminus(t):
    '''expression : MINUS expression %prec UMINUS'''
    # print("Inside unary minus")
    t[0] = -t[2]

def p_error(t):
    raise SyntaxError()
    # print("Syntax error at '%s'" % t.value)

parser = yacc.yacc()

def main():
    try:
        file = open(sys.argv[1], 'r')
    except:
        print("ERROR: The input file should be added as an argument while execution of the program.")
        exit()
    try:
        for line in file:
            try:
                result = parser.parse(line)
                print(result)
            except SemanticError as error:
                # print(error)
                print("SEMANTIC ERROR")
            except SyntaxError as error:
                # print(error)
                print("SYNTAX ERROR")
    finally:
        file.close()
        

if __name__ == "__main__":
    main()


# Citations for Reference:
# CSE 307 PLY Lecture Videos and Examples provided by Christopher Kane
# https://www.dabeaz.com/ply/ply.html#ply_nn24
# https://www.dabeaz.com/ply/example.html

