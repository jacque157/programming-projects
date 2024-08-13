class Literal:
    def __init__(self, positive, name, arguments):
        self.positive = positive
        self.name = name
        self.arguments = arguments

    def __str__(self):
        return f"{self.name}({','.join(self.arguments)})" if self.positive \
               else f"-{self.name}({','.join(self.arguments)})"

    def negate(self):
        return Literal(not self.positive, self.name, self.arguments)

    def __eq__(self, other):
        return self.name == other.name \
               and set(self.arguments) == set(other.arguments)\
               and self.positive == other.positive

    def at_step(self, step):
        return Literal(self.positive, self.name, (*self.arguments, str(step),))

    def __hash__(self):
        return hash(str(self))

class At(Literal):
    def __init__(self, where):
        super().__init__(True, 'at', (where, ))

    def negate(self):
        at = At(self.arguments[0])
        at.positive = not self.positive
        return at

class Box(Literal):
    def __init__(self, where):
        super().__init__(True, 'box', (where, ))

    def negate(self):
        box = Box(self.arguments[0])
        box.positive = not self.positive
        return box

class Wall(Literal):
    def __init__(self, where):
        super().__init__(True, 'wall', (where, ))

    def negate(self):
        wall = Wall(self.arguments[0])
        wall.positive = not self.positive
        return wall

class Goal(Literal):
    def __init__(self, where):
        super().__init__(True, 'goal', (where, ))

    def negate(self):
        goal = Goal(self.arguments[0])
        goal.positive = not self.positive
        return goal


class Action(Literal):
    def __init__(self, positive, name, arguments, positive_precoditions, negative_precoditions, add_list, remove_list):
        super().__init__(positive, name, arguments)
        self.positive_precoditions = positive_precoditions
        self.negative_precoditions = negative_precoditions
        self.add_list = add_list
        self.remove_list = remove_list
        
        self.precons = [precon for precon in self.positive_precoditions]
        self.precons.extend([precon.negate() for precon in self.negative_precoditions])

        self.precon_fluents = set(self.positive_precoditions).union(self.negative_precoditions )

        self.list = [precon for precon in self.add_list]
        self.list.extend([precon.negate() for precon in self.remove_list])

        self.list_fluents = set(self.add_list).union(self.remove_list )

    def preconditions(self):
        return self.precons

    def str_preconditions(self):
        return str(Clause(self.preconditions()))
    
    def effects(self):
        return self.list

    def precondition_fluents(self):
        return self.precon_fluents

    def effect_fluents(self):
        return self.list_fluents

    def str_effects(self):
        return str(Clause(self.effects()))

class Move(Action):
    def __init__(self, from_, to):
        super().__init__(True, 'move', (from_, to),
                         [At(from_)],
                         [Box(to), Wall(to)],
                         [At(to)],
                         [At(from_)])

class Push(Action):
    def __init__(self, position, from_, to):
        super().__init__(True, 'push', (position, from_, to),
                         [At(position), Box(from_)],
                         [Box(to), Wall(to)],
                         [At(from_), Box(to)],
                         [At(position), Box(from_)])

class Wait(Action):
    def __init__(self):
        super().__init__(True, 'wait', (),
                         [],
                         [],
                         [],
                         [])

class Clause:
    def __init__(self, literals):
        self.literals = literals

    def __str__(self):
        return ' '.join(map(str, self.literals))
