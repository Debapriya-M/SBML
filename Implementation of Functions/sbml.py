import ply.lex as lex
import sys
import ply.yacc as yacc
import warnings as warnings

# dict_variable = {}

warnings.filterwarnings("ignore")
class SemanticError(Exception):
    pass

class Node:
    def __init__(self):
        # print("Inside __init__ block")
        self.value = 0

    def evaluate(self):
        # print("Inside __init__ block - evaluate")
        return self.value

    def execute(self):
        # print("Inside __init__ block - execute")
        return self.value

class NumberNode(Node):
    def __init__(self, v):
        super().__init__()
        # print("Inside Number node", v)
        if('.' in v or 'e' in v or 'E' in v):
            self.value = float(v)
        else:
            self.value = int(v)

    def execute(self):
        return self.value

class StringNode(Node):
    def __init__(self, s):
        super().__init__()
        self.value = str(s)

    def get_element(self, index):
        if index < 0:
            raise Exception("String index cannot be negative")
        return StringNode(self.value[index])

    def execute(self):
        # return '\'' + self.value + '\'' #Implementation for HW3 - strings should be enclosed in single quotes
        return self.value

# Initializing a global stack for pushing peeping and popping values
class MyStack:
    def __init__(self):
        self.stack = [{}]

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        return self.stack.pop()

    def peep(self, i = -1):
        return self.stack[i]

my_stack = MyStack()

class VariableNode(Node):
    def __init__(self, var):
        super().__init__()
        self.value = var
        # print("Variable value = ", self.value)

    def execute(self):
        return my_stack.peep()[self.value]

class BooleanNode(Node):
    def __init__(self, s):
        super().__init__()
        # print("Inside Boolean Node")
        if s == 'True' or s == True:
            self.value = True
        else:
            self.value = False

    def execute(self):
        return self.value

class UminusNode(Node):
    def __init__(self, v1):
        super().__init__()
        self.v1 = v1

    def execute(self):
        v1 = self.v1.execute()
        if (isinstance(v1, float) or isinstance(v1, int)):
            return -1 * v1
        else:
            raise SemanticError

class ComparisonNode(BooleanNode):
    def __init__(self, comparator, v1, v2):
        super().__init__(False)
        # print("Inside ComparisonNode init")
        self.comparator = comparator
        self.v1 = v1
        self.v2 = v2
        # print("v1 : ", v1)
        # print("v2 : ", v2)
        # print("comparator : ", comparator)

    def execute(self):
        # print("Inside ComparisonNode execute")
        v1 = self.v1.execute()
        # print("v1: ", v1)
        v2 = self.v2.execute()
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
            # print("Inside < operator")
            self.value = (v1 < v2)
            return self.value
        elif self.comparator == '>':
            # print("Inside > operator")
            self.value = (v1 > v2)
            return self.value
        elif self.comparator == '<=':
            # print("Inside <= operator")
            self.value = (v1 <= v2)
            return self.value
        elif self.comparator == '>=':
            # print("Inside >= operator")
            self.value = (v1 >= v2)
            return self.value
        elif self.comparator == '==':
            # print("Inside == operator")
            self.value = (v1 == v2)
            return self.value
        elif self.comparator == '<>':
            # print("Inside <> operator")
            self.value = (v1 < v2 or v1 > v2)
            return self.value
        # print("+++++",self.value)
        else:
            # print("SEMANTIC Error right here - 1")
            raise SemanticError

class BooleanOpNode(BooleanNode):
    def __init__(self, comparator, v1, v2):
        super().__init__(False)
        self.comparator = comparator
        self.v1 = v1
        self.v2 = v2

    def execute(self):
        # print("Inside the boolean operator node")
        v1 = self.v1.execute()
        v2 = self.v2.execute()
        # print(v1)
        # print(v2)
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

class BopNode(Node):
    def __init__(self, op, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.op = op
        self.value = 0
        
    def typecheck(self):
        v1 = self.v1.execute()
        v2 = self.v2.execute()
        # print("v1: ", v1)
        # print("v2: ", v2)
        if type(v1) != int and type(v1) != float:
            raise SemanticError()
        elif type(v2) != int and type(v2) != float:
            raise SemanticError()
        else:
            pass

    def execute(self):
        # print("Inside BopNode execute")
        v1 = self.v1.execute()
        v2 = self.v2.execute()
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

class ListNode(Node):
    def __init__(self, v=None):
        super().__init__()
        if v is None:
            self.value = []
        else:
            self.value = [v]

    def append(self, v):
        self.value.append(v)
        return self

    def get_element(self, index):
        # print("Inside ListNode get element")
        if index < 0:
            raise Exception
        return self.value[index]

    def execute(self):
        return [x.execute() for x in self.value]

class TupleNode(Node):
    def __init__(self, l):
        super().__init__()
        # print("Inside TupleNode")
        self.value = tuple(l)

    def get_element(self, index):
        if index <= 0:
            raise Exception("SEMANTIC ERROR")
        return self.value[index - 1]

    def execute(self):
        return tuple([x.execute() for x in self.value])

class PrintNode(Node):
    def __init__(self, v):
        super().__init__()
        self.value = v

    def execute(self):
        # self.value = self.value.execute
        # print("print node execute = ", self.value)
        print(self.value.execute())

def storing_function(id, value, index = None):
    # print("Inside storing_function")
    temp_var = value.execute()
    if index is None:
        my_stack.peep()[id] = temp_var
    else:
        if isinstance(id, Node):
            id.execute()[index] = temp_var
        else:
            my_stack.peep()[id][index] = temp_var

class AssignmentNode(Node):
    def __init__(self, v, value, index = None):
        super().__init__()
        self.v = v.value
        self.value = value
        self.index = index
        # print("AssignmentNode value = ", self.v)

    def execute(self):
        index = self.index
        if index is not None:
            index = index.execute()
        storing_function(self.v, self.value, index)

class BlockNode(Node):
    def __init__(self, sl):
        super().__init__()
        self.statementList = sl

    def execute(self):
        if (self.statementList == None):
            None
        else:
            for statement in self.statementList:
                statement.execute()

#Implementation for finding the index of the list.
class IndexListNode(Node):
    def __init__(self, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2

    def execute(self):
        v1 = self.v1.execute()
        v2 = self.v2.execute()
        if ((isinstance(v1, list) or isinstance(v1, str)) and isinstance(v2, int)):
            return v1[v2]
        else:
            raise SemanticError

#Implementation for finding the index of the tuple.
class IndexTupleNode(Node):
    def __init__(self, v, index):
        super().__init__()
        self.v = v
        self.index = index
        self.value = 0

    def execute(self):
        # print("Inside TupleNode execute")
        index = self.index.execute()
        # print(index)
        v = self.v.execute()
        # print(v)
        if index <= 0:
            raise SemanticError
        self.value = v[index - 1]
        # print(self.value)
        return self.value

class IsMemberNode(Node):
    def __init__(self, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2

    def execute(self):
        v1 = self.v1.execute()
        v2 = self.v2.execute()
        # print("Inside IsMemberNode execute")
        if type(v2) != type("") and type(v2) != type([]):
            raise SemanticError()
        self.value = v1 in v2
        return self.value

class ConcatNode(Node):
    def __init__(self, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2

    def execute(self):
        v1 = self.v1.execute()
        v2 = self.v2.execute()
        if (isinstance(v2, list)):
            return [v1] + v2
        else:
            raise SemanticError

class NotOpNode(Node):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def execute(self):
        v = self.v.execute()
        if (isinstance(v, bool)):
            return not v
        else:
            raise SemanticError

class ConditionNode(Node):
    def __init__(self, condition, block, block2=None):
        super().__init__()
        self.condition = condition
        self.block = block
        self.block2 = block2

    def call_else(self, block):
        # print("Inside seting else block")
        self.block2 = block

    def execute(self):
        condition = self.condition.execute()
        if not (isinstance(condition, bool) and isinstance(self.block, BlockNode)):
            raise ValueError("neither boolean nor block")
        if condition:
            self.block.execute()
        else:
            if self.block2 is not None:
                self.block2.execute()

class WhileNode(Node):
    def __init__(self, condition, block):
        # print("Inside while node")
        super().__init__()
        self.condition = condition
        self.block = block

    def execute(self):
        condition = self.condition.execute()
        if not (isinstance(condition, bool) and isinstance(self.block, BlockNode)):
            raise ValueError("neither boolean nor block")
        while self.condition.execute():
            self.block.execute()

#Implementation of the Program Node
class ProgramNode(Node):
    def __init__(self, block, functions = None):
        super().__init__()
        self.block = block
        self.functions = functions

    def execute(self):
        if self.functions is not None:
            self.functions.execute()
        self.block.execute()

#Implementation of the function and storing this function on the global stack
class FunctionNode(Node):
    def __init__(self, v1, v2, block, output):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.block = block
        self.output = output

    def execute(self):
        v1 = self.v1.value
        my_stack.peep()[v1] = self

    def add_function(self, argv):
        temp_dic = {self.v1: self}
        # print(temp_dic)
        for i in range(len(argv)):
            arg_val = self.v2.get_element(i).value
            ex = argv[i].execute()
            temp_dic[arg_val] = ex
        #adding to the global stack
        my_stack.push(temp_dic)
        self.block.execute()

        res = self.output.execute()
        temp_dic =  my_stack.pop()
        return res

class FuncCallNode(Node):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def execute(self):
        v1 = self.v1.value
        v2 = self.v2.value
        self.value = my_stack.peep(0)[v1].add_function(v2)
        return self.value

reserved = {
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'print': 'PRINT',
    'mod': 'MOD',
    'div': 'DIV',
    'in': 'IN',
    'not': 'NOT',
    'andalso': 'ANDALSO',
    'orelse': 'ORELSE',
    'fun': 'FUN',
}

tokens = [
            'LPAREN', 'RPAREN','NUMBER', 'BOOLEAN',
            'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 
            'LBRACKET', 'RBRACKET', 'STRING', 'COMMA', 
            'TUPLE_INDEX', 'SEMICOLON', 'POWER', 'CONS',
            'LESS', 'GREATER', 'LESSEQUAL', 'GREATEREQUAL', 'EQUALEQUAL', 'NOTEQUAL',
            'VARIABLE', 'ASSIGN',
            'LCURLY', 'RCURLY',
         ] 

tokens = tokens + list(reserved.values())

# Tokens
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LCURLY = r'\{'
t_RCURLY = r'\}'
t_COMMA = r','
t_SEMICOLON = r';'
t_TUPLE_INDEX = r'[#]'
t_POWER = r'\*\*'
t_CONS = r'::'
t_EQUALEQUAL = r'=='
t_LESS = r'<'
t_GREATER = r'>'
t_LESSEQUAL = r'<='
t_GREATEREQUAL = r'>='
t_NOTEQUAL = r'<>'
t_ASSIGN = r'='


def t_NUMBER(t):
    r'-?\d*(\d\.|\.\d|\d)\d*[Ee](-|\+)?\d+|\d*(\d\.|\.\d)\d*|\d+'
    try:
        t.value = NumberNode(t.value)
    except ValueError:
        raise SyntaxError
    return t


def t_STRING(t):
    r'(\"([^\\\n]|(\\.))*?\") | (\'([^\\\n]|(\\.))*?\')'
    # print("Inside t_STRING")
    # print(t)
    try:
        t.value = StringNode(t.value[1:-1])
    except ValueError:
        raise SyntaxError
    return t


def t_BOOLEAN(t):
    r'(True)|(False)'
    t.value = BooleanNode(t.value)
    return t


def t_VARIABLE(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'VARIABLE')  # Check for reserved words
    if t.type == 'VARIABLE':
        t.value = VariableNode(t.value)
    return t

# Ignored characters
t_ignore = " \t"


def t_error(t):
    raise SyntaxError("SYNTAX ERROR_token not found", t)


# Build the lexer
lex.lex()

# Parsing rules
precedence = (
    ('left', 'ORELSE'),
    ('left', 'ANDALSO'),
    ('left', 'NOT'),
    ('left', 'LESS', 'GREATER', 'LESSEQUAL', 'GREATEREQUAL', 'EQUALEQUAL', 'NOTEQUAL'),
    ('right', 'CONS'),
    ('left', 'IN'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE', 'MOD', 'DIV'),
    ('right', 'UMINUS'),
    ('right', 'POWER'),
    ('nonassoc', 'LBRACKET', 'RBRACKET'),
    ('left', 'TUPLE_INDEX'),
    ('nonassoc', 'LPAREN', 'RPAREN'),
    ('left', 'IF', 'WHILE', 'ELSE'),
    ('left', 'LCURLY', 'RCURLY')
)


def p_program(p):
    '''program : functions block
               | block'''
    # print("Inside p_program")
    if len(p) > 2:
        p[0] = ProgramNode(p[2], p[1])
    else:
        p[0] = ProgramNode(p[1])

def p_program_funcs(p):
    '''functions : functions function
                 | function'''
    # print("Inside p_functions")
    if len(p) > 2:
        t[0] = t[1] + [t[2]]
    else:
        p[0] = ListNode(p[1])

def p_function(p):
    '''function : FUN VARIABLE LPAREN elements RPAREN ASSIGN block expression SEMICOLON'''
    # print("Inside p_function")
    p[0] = FunctionNode(p[2], p[4], p[7], p[8])


def p_id_elements(t):
    '''elements : elements COMMA VARIABLE
                | VARIABLE'''
    # print("Inside p_params")
    if len(t) > 2:
        t[1].append(t[3])
        t[0] = t[1]
    else:
        t[0] = ListNode(t[1])

def p_expression_func_call(p):
    '''expression : VARIABLE LPAREN elements RPAREN'''
    # print("Inside p_function_call")    
    p[0] = FuncCallNode(p[1],p[3])

def p_elements(t):
    '''elements : elements COMMA expression
                | expression'''
    # print("Inside p_arguments")
    if len(t) > 2:
        t[1].append(t[3])
        t[0] = t[1]
    else:
        t[0] = ListNode(t[1])

def p_block(p):
    '''block : LCURLY block RCURLY'''
    p[0] = p[2]

def p_block2(p):
    '''block : LCURLY statement_list RCURLY'''
    p[0] = BlockNode(p[2])

def p_empty_block(p):
    '''block : LCURLY RCURLY'''
    p[0] = BlockNode([])

def p_statement_list(p):
    '''statement_list : statement_list statement'''
    p[0] = p[1] + [p[2]]

def p_statement_list_val(p):
    '''statement_list : statement'''
    p[0] = [p[1]]

def p_statements(t):
    '''statement : block
                 | assignment SEMICOLON
                 | print SEMICOLON
                 | expression SEMICOLON
                 | ifelse_statement
                 | if_statement
                 | while_statement'''
    # print("Inside p_statement")
    t[0] = t[1]

def p_if_statement(p):
    '''if_statement : IF LPAREN expression RPAREN block'''
    # print("Inside p_statement_if")
    p[0] = ConditionNode(p[3], p[5])

def p_if_else_statement(p):
    '''ifelse_statement : if_statement ELSE block'''
    # print("Inside p_if_else_statement")
    p[1].call_else(p[3])
    p[0] = p[1]

def p_while(p):
    '''while_statement : WHILE LPAREN expression RPAREN block'''
    # print("Inside p_while")
    p[0] = WhileNode(p[3], p[5])

def p_assignment(t):
    '''assignment : VARIABLE ASSIGN expression
                  | expression LBRACKET expression RBRACKET ASSIGN expression '''
    # print("Inside p_assignment ")
    if len(t) == 4:
        t[0] = AssignmentNode(t[1], t[3])
    else:
        t[0] = AssignmentNode(t[1], t[6], t[3])

def p_print_statement(t):
    '''print : PRINT LPAREN expression RPAREN'''
    # print("Inside p_print_statement:")
    t[0] = PrintNode(t[3])

def p_expression_group(t):
    '''expression : LPAREN expression RPAREN'''
    # print("Inside p_expression_group")
    t[0] = t[2]

def p_factor(t):
    '''factor : VARIABLE
              | NUMBER
              | BOOLEAN
              | STRING
              | tuple
              | list'''
    t[0] = t[1]

def p_expression_not(t):
    '''expression : NOT expression'''
    # print("Inside p_expression_not")
    t[0] = NotOpNode(t[2])

def p_expression_factor(t):
    '''expression : factor
                  | indexing'''
    # print("Inside p_expression_factor")
    # print("t[1] :", t[1])
    t[0] = t[1]

def p_expression_binop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression POWER expression
                  | expression MOD expression
                  | expression DIV expression'''
    # print("Inside p_expression_binop")
    t[0] = BopNode(t[2], t[1], t[3])

def p_expression_comparison(t):
    '''expression : expression LESS expression
                  | expression GREATER expression
                  | expression LESSEQUAL expression
                  | expression GREATEREQUAL expression
                  | expression EQUALEQUAL expression
                  | expression NOTEQUAL expression'''
    # print("Inside p_expression_comparison")
    # print("t[1] : ", t[1])
    # print("t[2] : ", t[2])
    # print("t[3] : ", t[3])
    t[0] = ComparisonNode(t[2], t[1], t[3])

def p_expression_boolean_operation(t):
    '''expression : expression ANDALSO expression
                  | expression ORELSE expression'''
    t[0] = BooleanOpNode(t[2], t[1], t[3])

def p_expression_in(t):
    '''expression : expression IN expression'''
    # print("Inside p_expression_in")
    t[0] = IsMemberNode(t[1], t[3])

def p_expression_uminus(t):
    '''expression : MINUS expression %prec UMINUS'''
    # print("Inside p_expression_uminus")
    t[0] = UminusNode(t[2])

def p_expression_concatenation(t):
    '''expression : expression CONS expression'''
    # print("Inside p_expression_concatenation")
    t[0] = ConcatNode(t[1], t[3])

def p_expression_list(t):
    '''list : LBRACKET elements RBRACKET
            | LBRACKET RBRACKET'''
    # print("Inside p_expression_list")
    if len(t) > 3:
        # print("p_expression_list if")
        t[0] = t[2]
    else:
        # print("p_expression_list else")
        t[0] = ListNode()

def p_expression_list_index(t):
    '''indexing : expression LBRACKET expression RBRACKET'''
    # print("Inside p_expression_list_index")
    t[0] = IndexListNode(t[1], t[3])

def p_expression_tuple(t):
    '''tuple : LPAREN elements RPAREN
             | LPAREN RPAREN'''
    # print("Inside p_expression_tuple")
    if len(t) > 3:
        t[0] = TupleNode(t[2].value)
    else:
        t[0] = TupleNode([])

def p_expression_tuple_index(t):
    '''indexing : TUPLE_INDEX expression LPAREN expression RPAREN
                | TUPLE_INDEX expression expression '''
    # print("Inside p_expression_tuple_index")
    index = t[2]
    if len(t) > 4:
        t[0] = IndexTupleNode(t[4], index)
    else:
        t[0] = IndexTupleNode(t[3], index)

def p_error(t):
    raise SyntaxError("SYNTAX ERROR pattern not found at '%s'" % t.value)
    # print("Syntax error at '%s'" % t.value)

yacc.yacc()

def main():
    #checking whether the arguments given while running the python file is correct
    if (len(sys.argv) != 2):
        sys.exit("Invalid arguments! Please check the arguments.")
        
    try:
        #Reading the file as a complete string instead of reading it line by line, by replacing new lines with blanks.
        with open(sys.argv[1], 'r') as myfile:
            data = myfile.read().replace('\n', '')
        # print(data)
        root = yacc.parse(data)
        #calling the execute methods.
        root.execute()
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

