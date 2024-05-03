from lexer_parser import Lexer, Parser
from tensorflow_executor import TensorFlowExecutor
from torch_executor import PyTorchExecutor

class AIModelFramework:
    def __init__(self, backend):
        self.backend = backend
        if self.backend == "tensorflow":
            self.executor = TensorFlowExecutor()
        elif self.backend == "pytorch":
            self.executor = PyTorchExecutor()
        else:
            raise ValueError("Unsupported backend. Please choose either 'tensorflow' or 'pytorch'.")

    def build_and_execute_model(self, model_definition):
        # Tokenize and parse the model definition
        lexer = Lexer()
        parser = Parser()
        tokens = lexer.tokenize(model_definition)
        parsed_model = parser.parse(tokens)

        # Execute the parsed model using the appropriate executor
        self.executor.execute(parsed_model)
