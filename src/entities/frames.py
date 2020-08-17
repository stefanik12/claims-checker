from src.entities.token import TokenNode, Token


class NounFrame:

    def __init__(self, noun: Token, verb: Token, paragraph_i: int = None, context_i: int = None):
        self.paragraph_i = paragraph_i
        self.noun = noun
        self.closest_verb = verb


