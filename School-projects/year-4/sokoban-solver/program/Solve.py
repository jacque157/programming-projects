import subprocess
from Literals import *
from Plan import Plan

class Solver:
    MINI_SAT = 'minisat/MiniSat_v1.14.exe'
    OUTPUT = 'result'
    CNF = 'cnf'
    DIMAC = 'dimac'

    def write_model(dimac_path, output_path):
        cmd = f'{Solver.MINI_SAT} {dimac_path} {output_path}'
        return subprocess.call(cmd)


    def write_cnf(clauses, cnf_path):
        with open(cnf_path, 'w') as file:
            print('\n'.join(map(str, clauses)), file=file)
            """for clause in clauses:
                print(clause, file=file)"""

    def write_dimac(cnf_path, dimac_path):
        cnf_clauses = Solver.file_to_list(cnf_path)

        i = 1
        dimac_literals = Solver.dimac_literals_table(cnf_path)
        dimac_clauses = []
        for clause in cnf_clauses:
            dimac_clause = []
            if clause[0] == 'c':
                continue
            for literal in clause.strip().split():
                if literal[0] == '-':
                    literal = literal[1:]
                    dimac_clause.append(f'-{dimac_literals[literal]}')       
                else:
                    dimac_clause.append(f'{dimac_literals[literal]}')   
            dimac_clauses.append(' '.join(dimac_clause + ['0']))

        with open(dimac_path, 'w') as file:
            print('\n'.join(dimac_clauses), file=file)
            """for clause in dimac_clauses:
                print(clause, file=file)"""

    def file_to_list(path):
        output = []
        with open(path) as file:
            output = list(file)
            """for line in file:
                output.append(line)"""
        return output

    def dimac_literals_table(cnf_clauses_path):
        cnf_clauses = Solver.file_to_list(cnf_clauses_path)
        i = 1
        dimac_literals = {}
        for clause in cnf_clauses:
            dimac_clause = []
            if clause[0] == 'c':
                continue
            for literal in clause.strip().split():
                if literal[0] == '-':
                    literal = literal[1:]
                    
                if literal not in dimac_literals:
                    dimac_literals[literal] = str(i)
                    i += 1

        return dimac_literals 
                
    def read_model(model_path):
        with open(model_path) as file:
            sat = file.readline()
            if sat == 'UNSAT\n':
                return []
            model = file.readline().split()
        return model

    def decode_answer(dimac_answer_path, cnf_clauses_path, dimac_literals):
        dimac_answer = Solver.read_model(dimac_answer_path)
        dimac_literals = Solver.dimac_literals_table(cnf_clauses_path)
        answer = []
        literals = {}
        for lit, dim_lit in dimac_literals.items():
            literals[dim_lit] = lit 
        for lit in dimac_answer:
            if lit[0] == '-':
                continue
            if lit == '0':
                break
            literal = literals[lit]
            if literal.startswith('push') or literal.startswith('move'):
                answer.append(literal)
        return answer

    def parse_answer(answer):
        sequence = []
        for action in answer:
            start = action.find('(')
            if start == -1:
                continue
            name = action[:start]
            args = action[start:]

            if name == 'push':
                pos1, pos2, pos3, step = eval(args)
                sequence.append((step, pos1, pos2))
            if name == 'move':
                pos1, pos2, step = eval(args)
                sequence.append((step, pos1, pos2))
        return [Solver.get_direction(*pos1, *pos2) for _, pos1, pos2 in sorted(sequence, key=lambda x : x[0])]

    def get_direction(row1, col1, row2, col2):
        if row1 == row2:
            if col1 + 1 == col2:
                return 'Right'
            elif col1 - 1 == col2:
                return 'Left'
        if col1 == col2:
            if row1 - 1 == row2:
                return 'Up'
            elif row1 + 1 == row2:
                return 'Down'
        return None

    def find_solution(level_path, name=''):
        p = Plan(level_path)
        if name:
            cnf_name = f'{name}_cnf'
            dimacs_name = f'{name}_dimacs'
            solution_name = f'{name}_answer'
        else:
            cnf_name = 'debug_cnf'
            dimacs_name = 'debug_dimacs'
            solution_name = 'debug_answer'
            
        best_cnf_path = ''
        best_dimacs_path = ''
        best_answer_path = ''
        answer = []
        step = 4
        
        while True:
            cnf_path = f'cnf/{cnf_name}_{step}.txt'
            dimacs_path = f'dimacs/{dimacs_name}_{step}.txt'
            answer_path = f'answer/{solution_name}_{step}.txt'

            clauses = p.build_theory(step)
            Solver.write_cnf(clauses, cnf_path)
            Solver.write_dimac(cnf_path, dimacs_path)
            Solver.write_model(dimacs_path, answer_path)

            answer = Solver.read_model(answer_path)
            if answer:
                best_cnf_path = cnf_path
                best_dimacs_path = dimacs_path
                best_answer_path = answer_path
                solution_path = answer_path
                if step > 4:
                    break
                step //= 2
            else:
                if best_answer_path:
                    break
                step *= 2
        return Solver.decode_answer(best_answer_path, best_cnf_path, best_dimacs_path)               
