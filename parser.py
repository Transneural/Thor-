from ply import yacc
from lexer import Lexer

class Parser:
 def __init__(self, Lexer):
        self.lexer = Lexer
        self.tokens = lexer.tokens 
        
 def p_statement(p):
    '''statement : neural_network_definition
                 | loss_function_definition
                 | optimizer_definition
                 | automatic_differentiation
                 | function_definition
                 | expression_statement
                 | assignment_statement
                 | return_statement
                 | if_statement
                 | switch_case_statement
                 | ternary_operator
                 | try_catch_block
                 | import_statement
                 | class_definition
                 | method_definition
                 | data_load_statement
                 | preprocess_statement
                 | evaluate_statement
                 | metrics_statement
                 | function_call_statement
                 | math_functions
                 | array_functions
                 | file_operations
                 | generate_layer_statement  
                 | prune_network_statement    
                 | get_weights_statement 
                 | set_weights_statement  
                 | custom_activation_definition
                 | get_weights_statement
                 | set_weights_statement
                 | optimize_using_genetic_algorithm
                 | meta_optimize_statement    # Add meta_optimize_statement
                 | strategy_statement     # Add strategy_statement
                 | hybrid_reasoning
                 | data_load_statement(
                 '''
    p[0] = p[1]

# Update import statement to support importing classes and methods
 def p_import_statement(p):
    '''import_statement : IMPORT IDENTIFIER SEMICOLON
                        | IMPORT IDENTIFIER DOT IDENTIFIER SEMICOLON'''
    if len(p) == 4:
        p[0] = ('Import', p[2])
    else:
        p[0] = ('Import', p[2], p[4])

 def p_neural_network_definition(p):
    '''neural_network_definition : NEURAL_NETWORK IDENTIFIER LBRACKET layer_list RBRACKET SEMICOLON'''
    p[0] = ('NeuralNetwork', p[2], p[4])

 def p_layer(p):
    '''layer : layer_type LPAREN layer_params RPAREN'''
    p[0] = (p[1], p[3])

 def p_layer_type(p):
    '''layer_type : DENSE
                  | CONV2D
                  | MAXPOOLING2D
                  | FLATTEN'''
    p[0] = p[1]

 def p_layer_params(p):
    '''layer_params : layer_param
                    | layer_params COMMA layer_param'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_layer_param(p):
    '''layer_param : IDENTIFIER ASSIGN param_value
                   | ACTIVATION ASSIGN activation_function'''
    p[0] = (p[1], p[3])

 def p_param_value(p):
    '''param_value : INTEGER
                   | FLOAT
                   | IDENTIFIER'''
    p[0] = p[1]

 def p_loss_function_definition(p):
    '''loss_function_definition : LOSS_FUNCTION IDENTIFIER ASSIGN loss_function SEMICOLON'''
    p[0] = ('LossFunction', p[2], p[4])

 def p_loss_function(p):
    '''loss_function : MEAN_SQUARED_ERROR
                     | CATEGORICAL_CROSSENTROPY'''
    p[0] = p[1]
    
 def p_optimizer_definition(p):
    '''optimizer_definition : OPTIMIZER IDENTIFIER ASSIGN optimizer SEMICOLON'''
    p[0] = ('Optimizer', p[2], p[4])

 def p_optimizer(p):
    '''optimizer : SGD LPAREN learning_rate COMMA momentum RPAREN
                 | ADAM LPAREN learning_rate COMMA beta_1 COMMA beta_2 RPAREN'''
    p[0] = (p[1], p[3:])
    
 def p_automatic_differentiation(p):
    '''automatic_differentiation : AUTOMATIC_DIFFERENTIATION LPAREN expression RPAREN SEMICOLON'''
    p[0] = ('AutomaticDifferentiation', p[3])

 def p_function_definition(p):
    '''function_definition : FUNCTION IDENTIFIER LPAREN parameter_list RPAREN LBRACKET statement_list RBRACKET'''
    p[0] = ('Function', p[2], p[4], p[7])

 def p_meta_optimize_statement(p):
    '''meta_optimize_statement : META_OPTIMIZE expression SEMICOLON'''
    p[0] = ('MetaOptimize', p[2])
    
 def p_activation_function(p):
    '''activation_function : RELU
                            | SIGMOID
                            | SOFTMAX
                            | CUSTOM_ACTIVATION
                            | LEAKY_RELU'''  # Add new activation function
    if p[1] == 'custom_activation':
        p[0] = p[1]
    else:
        p[0] = p[1]

 def p_custom_activation(p):
    '''custom_activation : CUSTOM_ACTIVATION LPAREN expression RPAREN'''
    p[0] = ('CustomActivation', p[3])
    
 def p_parameter_list(p):
    '''parameter_list : IDENTIFIER
                      | parameter_list COMMA IDENTIFIER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_statement_list(p):
    '''statement_list : statement
                      | statement_list statement'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

 def p_expression_statement(p):
    '''expression_statement : expression SEMICOLON'''
    p[0] = ('ExpressionStatement', p[1])

 def p_expression(p):
    '''expression : LPAREN operation operand RPAREN'''
    p[0] = (p[2], p[3])

 def p_operation(p):
    '''operation : PLUS
                 | MINUS
                 | TIMES
                 | DIVIDE'''
    p[0] = p[1]

 def p_operand(p):
    '''operand : IDENTIFIER
               | INTEGER
               | FLOAT
               | expression'''
    p[0] = p[1]

 def p_assignment_statement(p):
    '''assignment_statement : IDENTIFIER ASSIGN expression SEMICOLON'''
    p[0] = ('Assignment', p[1], p[3])

 def p_return_statement(p):
    '''return_statement : RETURN expression SEMICOLON'''
    p[0] = ('Return', p[2])

 def p_if_statement(p):
    '''if_statement : IF LPAREN expression RPAREN LBRACKET statement_list RBRACKET
                    | IF LPAREN expression RPAREN LBRACKET statement_list RBRACKET ELSE LBRACKET statement_list RBRACKET'''
    if len(p) == 8:
        p[0] = ('IfStatement', p[3], p[6], None)
    else:
        p[0] = ('IfStatement', p[3], p[6], p[10])

 def p_switch_case_statement(p):
    '''switch_case_statement : SWITCH LPAREN expression RPAREN LBRACKET case_list default_case RBRACKET'''
    p[0] = ('SwitchCaseStatement', p[3], p[6], p[7])

 def p_case_list(p):
    '''case_list : CASE expression COLON statement_list case_list
                 | empty'''
    if len(p) == 2:
        p[0] = []
    else:
        p[0] = [(p[2], p[4])] + p[5]

 def p_default_case(p):
    '''default_case : DEFAULT COLON statement_list
                    | empty'''
    if len(p) == 2:
        p[0] = []
    else:
        p[0] = p[3]

 def p_ternary_operator(p):
    '''ternary_operator : expression TERNARY expression COLON expression'''
    p[0] = ('TernaryOperator', p[1], p[3], p[5])

 def p_try_catch_block(p):
    '''try_catch_block : TRY LBRACKET statement_list RBRACKET CATCH LPAREN IDENTIFIER RPAREN LBRACKET statement_list RBRACKET'''
    p[0] = ('TryCatchBlock', p[3], p[10], p[7])

 def p_function_call_statement(p):
    '''function_call_statement : FUNCTION_CALL IDENTIFIER LPAREN arg_list RPAREN SEMICOLON'''
    p[0] = ('FunctionCall', p[2], p[4])

 def p_arg_list(p):
    '''arg_list : expression
                | arg_list COMMA expression'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_math_functions(p):
    '''math_functions : SIN LPAREN expression RPAREN SEMICOLON
                      | COS LPAREN expression RPAREN SEMICOLON
                      | EXP LPAREN expression RPAREN SEMICOLON
                      | LOG LPAREN expression RPAREN SEMICOLON'''
    p[0] = ('MathFunction', p[1], p[3])

 def p_array_functions(p):
    '''array_functions : RESHAPE LPAREN expression COMMA expression RPAREN SEMICOLON
                       | CONCATENATE LPAREN expression_list COMMA expression RPAREN SEMICOLON
                       | SPLIT LPAREN expression COMMA expression COMMA expression RPAREN SEMICOLON'''
    p[0] = ('ArrayFunction', p[1], p[3:])

 def p_expression_list(p):
    '''expression_list : expression
                       | expression_list COMMA expression'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_file_operations(p):
    '''file_operations : READ_FILE LPAREN expression RPAREN SEMICOLON
                       | WRITE_FILE LPAREN expression COMMA expression RPAREN SEMICOLON
                       | LIST_FILES LPAREN expression RPAREN SEMICOLON'''
    p[0] = ('FileOperation', p[1], p[3:])

 def p_data_load_statement(p):
    '''data_load_statement : DATA_LOAD LPAREN STRING_LITERAL COMMA IDENTIFIER RPAREN SEMICOLON'''
    p[0] = ('DataLoad', p[3], p[5])

 def p_preprocess_statement(p):
    '''preprocess_statement : PREPROCESS LPAREN IDENTIFIER COMMA IDENTIFIER RPAREN SEMICOLON'''
    p[0] = ('Preprocess', p[3], p[5])

 def p_evaluate_statement(p):
    '''evaluate_statement : EVALUATE LPAREN IDENTIFIER COMMA IDENTIFIER RPAREN SEMICOLON'''
    p[0] = ('Evaluate', p[3], p[5])

 def p_metrics_statement(p):
    '''metrics_statement : METRICS LPAREN IDENTIFIER COMMA IDENTIFIER RPAREN SEMICOLON'''
    p[0] = ('Metrics', p[3], p[5])


 def p_empty(p):
    'empty :'
    pass

 def p_generate_layer_statement(p):
    '''generate_layer_statement : GENERATE_LAYER condition LBRACKET layer RBRACKET SEMICOLON'''
    p[0] = ('GenerateLayer', p[2], p[4])

 def p_condition(p):
    '''condition : IF LPAREN expression RPAREN'''
    p[0] = p[3]

# Rules for pruning network
 def p_prune_network_statement(p):
    '''prune_network_statement : PRUNE_NETWORK LPAREN expression RPAREN SEMICOLON'''
    p[0] = ('PruneNetwork', p[3])
    
 def p_get_weights_statement(p):
    '''get_weights_statement : IDENTIFIER DOT IDENTIFIER LBRACKET INTEGER RBRACKET DOT GET_WEIGHTS LPAREN RPAREN SEMICOLON'''
    p[0] = ('GetWeights', p[1], p[3], p[5])

# Rules for set_weights
 def p_set_weights_statement(p):
    '''set_weights_statement : IDENTIFIER DOT IDENTIFIER LBRACKET INTEGER RBRACKET DOT SET_WEIGHTS LPAREN expression RPAREN SEMICOLON'''
    p[0] = ('SetWeights', p[1], p[3], p[5], p[10])
    
 def p_custom_activation_definition(p):
    '''custom_activation_definition : CUSTOM_ACTIVATION'''
    p[0] = ('CustomActivation', p[1])

# New rule for optimizer definition with constraints
 def p_optimizer_definition(p):
    '''optimizer_definition : OPTIMIZER IDENTIFIER LPAREN learning_rate ASSIGN FLOAT RPAREN WITH constraint SEMICOLON'''
    p[0] = ('Optimizer', p[2], p[6], p[9])

 def p_constraint(p):
    '''constraint : IDENTIFIER LESS_THAN FLOAT
                  | IDENTIFIER GREATER_THAN FLOAT'''
    p[0] = (p[1], p[2], p[3])

# New rule for genetic algorithm
 def t_GENETICALGORITHM(t):
    r'geneticalgorithm'
    return t

# Define parsing rules
 def p_optimize_using_genetic_algorithm(p):
    '''optimize_using_genetic_algorithm : OPTIMIZE expression USING GENETICALGORITHM SEMICOLON'''
    p[0] = ('OptimizeUsingGeneticAlgorithm', p[2])

 def p_expression(p):
    '''expression : IDENTIFIER
                  | INTEGER
                  | FLOAT
                  | LPAREN operation operand RPAREN'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = (p[2], p[3])


 def p_strategy_statement(p):
    '''strategy_statement : STRATEGY algorithm SEMICOLON'''
    p[0] = ('Strategy', p[2])

 def p_algorithm(p):
    '''algorithm : BAYESIAN
                 | REINFORCEMENT
                 | EVOLUTIONARY
                 | DYNAMIC_LR'''
    p[0] = p[1]
    

 def p_knowledge_representation(p):
    '''knowledge_representation : fact_definition
                                 | rule_definition'''
    p[0] = p[1]

 def p_fact_definition(p):
    '''fact_definition : FACT predicate SEMICOLON'''
    p[0] = ('Fact', p[2])

 def p_rule_definition(p):
    '''rule_definition : RULE predicate IF condition THEN action SEMICOLON'''
    p[0] = ('Rule', p[2], p[4], p[6])

 def p_action(p):
    '''action : ACTIVATE IDENTIFIER
              | ADJUST_THRESHOLD IDENTIFIER INTEGER'''
    if len(p) == 3:
        p[0] = ('Activate', p[2])
    else:
        p[0] = ('AdjustThreshold', p[2], p[3])

 def p_condition(p):
    '''condition : atomic_condition
                 | condition AND condition
                 | condition OR condition
                 | NOT condition'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 3:
        p[0] = ('Not', p[2])
    else:
        p[0] = (p[2], p[1], p[3])

 def p_atomic_condition(p):
    '''atomic_condition : predicate'''
    p[0] = p[1]

 def p_predicate(p):
    '''predicate : IDENTIFIER LPAREN predicate_arguments RPAREN'''
    p[0] = (p[1], p[3])

 def p_predicate_arguments(p):
    '''predicate_arguments : predicate_argument
                           | predicate_arguments COMMA predicate_argument'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_predicate_argument(p):
    '''predicate_argument : STRING_LITERAL
                          | IDENTIFIER'''
    p[0] = p[1]

    
 def p_action_activate(p):
    '''action : ACTIVATE IDENTIFIER INTEGER'''
    p[0] = ('Activate', p[2], p[3])

 def p_action_adjust_threshold(p):
    '''action : ADJUST_THRESHOLD IDENTIFIER INTEGER INTEGER'''
    p[0] = ('AdjustThreshold', p[2], p[3], p[4])

 def p_action_recall(p):
    '''action : RECALL IDENTIFIER'''
    p[0] = ('Recall', p[2])

 def p_evolutionary_operations(p):
    '''evolutionary_operations : crossover_statement
                                | mutate_statement'''
    p[0] = p[1]

 def p_crossover_statement(p):
    '''crossover_statement : CROSSOVER layer LPAREN IDENTIFIER RPAREN WITH layer LPAREN IDENTIFIER RPAREN STORE_AS IDENTIFIER SEMICOLON'''
    p[0] = ('Crossover', p[4], p[7], p[10])

 def p_mutate_statement(p):
    '''mutate_statement : MUTATE layer LPAREN IDENTIFIER RPAREN RATE FLOAT SEMICOLON'''
    p[0] = ('Mutate', p[4], p[7])
    
 def p_hybrid_reasoning(p):
    '''hybrid_reasoning : HYBRID_REASONING LPAREN expression COMMA knowledge_representation RPAREN SEMICOLON'''
    p[0] = ('HybridReasoning', p[3], p[5])

 def p_data_load_statement(p):
    '''data_load_statement : DATA_LOAD LPAREN STRING_LITERAL COMMA IDENTIFIER RPAREN SEMICOLON'''
    p[0] = ('DataLoad', p[3], p[5])
    
 def p_metrics_statement(p):
    '''metrics_statement : METRICS LPAREN IDENTIFIER COMMA IDENTIFIER RPAREN SEMICOLON'''
    p[0] = ('Metrics', p[3], p[5])

# Define parsing rules for training-related statements
 def p_training_statement(p):
    '''training_statement : TRAIN LPAREN training_config RPAREN SEMICOLON
                          | FINE_TUNE LPAREN training_config RPAREN SEMICOLON'''
    p[0] = ('Training', p[1], p[3])

 def p_training_config(p):
    '''training_config : EPOCHS ASSIGN INTEGER
                       | BATCH_SIZE ASSIGN INTEGER
                       | LEARNING_RATE ASSIGN FLOAT
                       | training_config COMMA training_config'''
    if len(p) == 4:
        p[0] = (p[1], p[3])
    else:
        p[0] = p[1] + p[3]

# Define parsing rules for search space
 def p_search_space_definition(p):
    '''search_space_definition : SEARCH_SPACE IDENTIFIER BLOCK_START search_space_content BLOCK_END SEMICOLON'''
    p[0] = ('SearchSpace', p[2], p[4])

 def p_search_space_content(p):
    '''search_space_content : search_space_block
                            | search_space_content search_space_block'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

 def p_search_space_block(p):
    '''search_space_block : choice_block
                         | combinator_block
                         | nested_block'''
    p[0] = p[1]

 def p_choice_block(p):
    '''choice_block : CHOICE IDENTIFIER LPAREN choice_options RPAREN SEMICOLON'''
    p[0] = ('ChoiceBlock', p[2], p[4])

 def p_choice_options(p):
    '''choice_options : option
                      | choice_options COMMA option'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_option(p):
    '''option : FLOAT
              | INTEGER
              | IDENTIFIER'''
    p[0] = p[1]

 def p_combinator_block(p):
    '''combinator_block : COMBINATOR IDENTIFIER LPAREN combinator_options RPAREN SEMICOLON'''
    p[0] = ('CombinatorBlock', p[2], p[4])

 def p_combinator_options(p):
    '''combinator_options : option
                          | combinator_options COMMA option'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_nested_block(p):
    '''nested_block : BLOCK_START search_space_content BLOCK_END'''
    p[0] = ('NestedBlock', p[2])
    
# Define parsing rules for meta-optimization statements
 def p_meta_optimization_statement(p):
    '''meta_optimization_statement : META_OPTIMIZE LPAREN optimization_technique COMMA search_space_identifier RPAREN SEMICOLON'''
    p[0] = ('MetaOptimization', p[3], p[5])

 def p_optimization_technique(p):
    '''optimization_technique : GENETIC_ALGORITHM
                               | REINFORCEMENT_LEARNING'''
    p[0] = p[1]

 def p_search_space_identifier(p):
    '''search_space_identifier : IDENTIFIER'''
    p[0] = p[1]
    
 def p_save_model_statement(p):
    '''save_model_statement : SAVE_MODEL LPAREN STRING_LITERAL RPAREN SEMICOLON'''
    p[0] = ('SaveModel', p[3])

 def p_load_model_statement(p):
    '''load_model_statement : LOAD_MODEL LPAREN STRING_LITERAL RPAREN SEMICOLON'''
    p[0] = ('LoadModel', p[3])

 def p_checkpoint_statement(p):
    '''checkpoint_statement : CHECKPOINT IDENTIFIER SEMICOLON'''
    p[0] = ('Checkpoint', p[2])

 def p_save_checkpoint_statement(p):
    '''save_checkpoint_statement : SAVE_CHECKPOINT IDENTIFIER LPAREN STRING_LITERAL RPAREN SEMICOLON'''
    p[0] = ('SaveCheckpoint', p[2], p[4])

 def p_load_checkpoint_statement(p):
    '''load_checkpoint_statement : LOAD_CHECKPOINT IDENTIFIER LPAREN STRING_LITERAL RPAREN SEMICOLON'''
    p[0] = ('LoadCheckpoint', p[2], p[4])

 def p_explanation_statement(p):
    '''explanation_statement : EXPLAIN IDENTIFIER FOR IDENTIFIER USING EXPLANATION_TECHNIQUE SEMICOLON'''
    p[0] = ('Explanation', p[2], p[4], p[6])

 def p_adaptation_statement(p):
    '''adaptation_statement : ADAPT IDENTIFIER WITH IDENTIFIER USING ADAPTATION_TECHNIQUE SEMICOLON'''
    p[0] = ('Adaptation', p[2], p[4], p[6])

# Define parsing rules for knowledge retention
 def p_training_process(p):
    '''training_process : REGULARIZATION
                        | EXPERIENCE_REPLAY'''
    p[0] = ('TrainingProcess', p[1])

# Define parsing rules for neuromorphic & spiking neural networks
 def p_neuromorphic_layer(p):
    '''neuromorphic_layer : NEUROMORPHIC_LAYER LPAREN neuromorphic_params RPAREN SEMICOLON'''
    p[0] = ('NeuromorphicLayer', p[3])

 def p_neuromorphic_params(p):
    '''neuromorphic_params : PARAMETER
                           | neuromorphic_params COMMA PARAMETER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_spiking_activation(p):
    '''spiking_activation : SPIKING_ACTIVATION LPAREN spiking_params RPAREN SEMICOLON'''
    p[0] = ('SpikingActivation', p[3])

 def p_spiking_params(p):
    '''spiking_params : PARAMETER
                      | spiking_params COMMA PARAMETER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_neuromorphic_layer(p):
      '''neuromorphic_layer : NEUROMORPHIC_LAYER LPAREN neuromorphic_params RPAREN SEMICOLON'''
      p[0] = ('NeuromorphicLayer', p[3])

 def p_neuromorphic_params(p):
      '''neuromorphic_params : PARAMETER
                              | neuromorphic_params COMMA PARAMETER'''
      if len(p) == 2:
         p[0] = [p[1]]
      else:
         p[0] = p[1] + [p[3]]

 def p_spiking_activation(p):
      '''spiking_activation : SPIKING_ACTIVATION LPAREN spiking_params RPAREN SEMICOLON'''
      p[0] = ('SpikingActivation', p[3])

 def p_spiking_params(p):
      '''spiking_params : PARAMETER
                        | spiking_params COMMA PARAMETER'''
      if len(p) == 2:
         p[0] = [p[1]]
      else:
         p[0] = p[1] + [p[3]]

 def p_temporal_dependency(p):
    '''temporal_dependency : TEMPORAL_DEPENDENCY LPAREN timing_params RPAREN SEMICOLON'''
    p[0] = ('TemporalDependency', p[3])

 def p_timing_params(p):
    '''timing_params : PARAMETER
                     | timing_params COMMA PARAMETER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

# Define parsing rules for symbolic NeuroAI
 def p_symbolic_operation(p):
    '''symbolic_operation : SYMBOLIC_OPERATION LPAREN symbolic_params RPAREN SEMICOLON'''
    p[0] = ('SymbolicOperation', p[3])

 def p_symbolic_params(p):
    '''symbolic_params : PARAMETER
                       | symbolic_params COMMA PARAMETER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

 def p_programmatic_architecture(p):
    '''programmatic_architecture : PROGRAMMATIC_ARCHITECTURE LPAREN architecture_params RPAREN SEMICOLON'''
    p[0] = ('ProgrammaticArchitecture', p[3])

 def p_architecture_params(p):
    '''architecture_params : PARAMETER
                           | architecture_params COMMA PARAMETER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

lexer = Lexer()
parser = yacc.yacc(module=Parser(lexer))
