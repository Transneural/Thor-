from ply import lex

class Lexer:
    # Define tokens
    tokens = (
        'IDENTIFIER',
        'INTEGER',
        'FLOAT',
        'RPAREN',
        'LBRACKET',
        'RBRACKET',
        'COMMA',
        'SEMICOLON',
        'ASSIGN',
        'PLUS',
        'MINUS',
        'TIMES',
        'DIVIDE',
        'ACTIVATION',
        'NEURAL_NETWORK',
        'LOSS_FUNCTION',
        'OPTIMIZER',
        'AUTOMATIC_DIFFERENTIATION',
        'FUNCTION',
        'RETURN',
        'IMPORT',
        'IF',
        'ELSE',
        'SWITCH',
        'CASE',
        'DEFAULT',
        'TERNARY',
        'TRY',
        'CATCH',
        'FUNCTION_CALL',
        'SIN',
        'COS',
        'EXP',
        'LOG',
        'RESHAPE',
        'CONCATENATE',
        'SPLIT',
        'READ_FILE',
        'WRITE_FILE',
        'LIST_FILES',
        'GENERATE_LAYER',
        'PRUNE_NETWORK',
        'GET_WEIGHTS',    
        'SET_WEIGHTS',
        'CUSTOM_ACTIVATION',
        'LESS_THAN',
        'GREATER_THAN',
        'GENETICALGORITHM',
        'META_OPTIMIZE',
        'STRATEGY',
        'BAYESIAN',
        'REINFORCEMENT',
        'EVOLUTIONARY',
        'DYNAMIC_LR',
        'FACT',
        'RULE',
        'AND',
        'OR',
        'NOT',
        'HYBRID_REASONING',
        'DATA_LOAD',
        'METRICS',
        'TRAIN',
        'FINE_TUNE',
        'EPOCHS',
        'BATCH_SIZE',
        'LEARNING_RATE',
        'SEARCH_SPACE',
        'BLOCK_START',
        'BLOCK_END',
        'CHOICE',
        'COMBINATOR',
        'LPAREN',
        'GENETIC_ALGORITHM',
        'REINFORCEMENT_LEARNING',
        'SAVE_MODEL',
        'LOAD_MODEL',
        'CHECKPOINT',
        'SAVE_CHECKPOINT',
        'LOAD_CHECKPOINT',
        'THEN',
        'ACTIVATE',
        'ADJUST_THRESHOLD',
        'RECALL',
        'EXPLAIN',
        'FOR',
        'EXPLANATION_TECHNIQUE',
        'ADAPT',
        'WITH',
        'USING',
        'ADAPTATION_TECHNIQUE',
        'REGULARIZATION',
        'EXPERIENCE_REPLAY',
        'NEUROMORPHIC_LAYER',
        'SPIKING_ACTIVATION',
        'TEMPORAL_DEPENDENCY',
        'SYMBOLIC_OPERATION',
        'PROGRAMMATIC_ARCHITECTURE'
    )

    # Regular expression rules for tokens
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_COMMA = r','
    t_SEMICOLON = r';'
    t_ASSIGN = r'='
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_SEARCH_SPACE = r'search_space'
    t_BLOCK_START = r'{'
    t_BLOCK_END = r'}'
    t_CHOICE = r'choice'
    t_COMBINATOR = r'combinator'
    t_META_OPTIMIZE = r'meta_optimize'
    t_GENETIC_ALGORITHM = r'genetic_algorithm'
    t_REINFORCEMENT_LEARNING = r'reinforcement_learning'

    # Regular expression rules with action code
    def t_ACTIVATION(t):
        r'activation\s+[a-zA-Z_][a-zA-Z0-9_]*'  # Example: activation relu
        t.value = t.value.split()[1]  # Extract activation function name
        return t
    
    def t_NEURAL_NETWORK(t):
     r'neural_network'
     return t

    def t_LOSS_FUNCTION(t):
     r'loss_function'
     return t

    def t_OPTIMIZER(t):
     r'optimizer'
     return t

    def t_AUTOMATIC_DIFFERENTIATION(t):
     r'automatic_differentiation'
     return t

    def t_IDENTIFIER(t):
     r'[a-zA-Z_][a-zA-Z0-9_]*'
     t.type = 'IDENTIFIER'
     return t

    def t_INTEGER(t):
     r'\d+'
     t.value = int(t.value)
     return t

    def t_FLOAT(t):
     r'\d+\.\d+'
     t.value = float(t.value)
     return t

    def t_IMPORT(t):
     r'import'
     return t

    def t_IF(t):
     r'if'
     return t

    def t_ELSE(t):
     r'else'
     return t

    def t_SWITCH(t):
     r'switch'
     return t

    def t_CASE(t):
     r'case'
     return t

    def t_DEFAULT(t):
     r'default'
     return t

    def t_TERNARY(t):
     r'\?'
     return t

    def t_TRY(t):
     r'try'
     return t

    def t_CATCH(t):
     r'catch'
     return t

    def t_FUNCTION_CALL(t):
     r'call'
     return t

    def t_SIN(t):
     r'sin'
     return t

    def t_COS(t):
     r'cos'
     return t

    def t_EXP(t):
     r'exp'
     return t

    def t_LOG(t):
     r'log'
     return t

    def t_RESHAPE(t):
     r'reshape'
     return t

    def t_CONCATENATE(t):
     r'concatenate'
     return t

    def t_SPLIT(t):
     r'split'
     return t

    def t_READ_FILE(t):
     r'read_file'
     return t

    def t_WRITE_FILE(t):
     r'write_file'
     return t

    def t_LIST_FILES(t):
     r'list_files'
     return t

    def t_GENERATE_LAYER(t):
     r'generate_layer'
     return t

    def t_PRUNE_NETWORK(t):
     r'prune_network'
     return t

    def t_error(t):
     print(f"Lexer Error at line {t.lineno}: Illegal character '{t.value[0]}'")
     t.lexer.skip(1)

    def t_UNDEFINED(t):
     r'.+'
     print(f"Lexer Error at line {t.lineno}: Undefined token '{t.value}'")
     t.lexer.skip(1)
    
    def t_GET_WEIGHTS(t):
     r'get_weights'
     return t

    def t_SET_WEIGHTS(t):
     r'set_weights'
     return t

    def t_CUSTOM_ACTIVATION(t):
     r'activation\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*{.*?}'
     t.value = t.value.strip()
     return t

    def t_LESS_THAN(t):
     r'<'
     return t

    def t_GREATER_THAN(t):
     r'>'
     return t
    
    def t_PARAMETER(t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        t.type = 'PARAMETER'
        return t

    def t_DATA_LOAD(t):
     r'data_load'
     return t

    def t_METRICS(t):
     r'metrics'  # Regular expression for METRICS token
     return t

    def t_TRAIN(t):
     r'train'
     return t
    
    def t_LEAKY_RELU(t):
        r'leaky_relu'
        return t

    def t_FINE_TUNE(t):
     r'fine_tune'
     return t

    def t_EPOCHS(t):
     r'epochs'
     return t

    def t_BATCH_SIZE(t):
     r'batch_size'
     return t

    def t_LEARNING_RATE(t):
     r'learning_rate'
     return t

    def t_SAVE_MODEL(t):
     r'save_model'
     return t

    def t_LOAD_MODEL(t):
     r'load_model'
     return t

    def t_CHECKPOINT(t):
     r'checkpoint'
     return t

    def t_SAVE_CHECKPOINT(t):
     r'save_checkpoint'
     return t

    def t_LOAD_CHECKPOINT(t):
     r'load_checkpoint'
     return t

    def t_THEN(t):
     r'then'
     return t

    def t_ACTIVATE(t):
     r'activate'
     return t

    def t_ADJUST_THRESHOLD(t):
     r'adjust_threshold'
     return t

    def t_RECALL(t):
     r'recall'
     return t

    def t_EXPLAIN(t):
     r'explain'
     return t

    def t_FOR(t):
     r'for'
     return t

    def t_EXPLANATION_TECHNIQUE(t):
     r'lime|shap|integrated_gradients'
     return t

    def t_ADAPT(t):
     r'adapt'
     return t

    def t_WITH(t):
     r'with'
     return t

    def t_USING(t):
     r'using'
     return t

    def t_ADAPTATION_TECHNIQUE(t):
     r'few_shot_learning|transfer_learning|domain_adaptation'
     return t


    def tokenize(self, data):
        self.lexer.input(data)
        tokens = []
        while True:
            tok = self.lexer.token()
            if not tok:
                break  # No more input
            tokens.append(tok)
        return tokens

    t_ignore = ' \t'


    lexer = lex.lex()