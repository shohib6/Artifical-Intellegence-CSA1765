class CSP:
    def __init__(self, variables, domains):  # Corrected the method name
        self.variables = variables
        self.domains = domains

    def is_consistent(self, variable, assignment):
        return all(assignment[neighbor] != assignment[variable] for neighbor in self.variables[variable] if neighbor in assignment)

    def backtracking_search(self, assignment={}):
        if len(assignment) == len(self.variables):
            return assignment
        
        unassigned = [var for var in self.variables if var not in assignment]
        first_unassigned = unassigned[0]
        
        for value in self.domains[first_unassigned]:
            assignment[first_unassigned] = value
            if self.is_consistent(first_unassigned, assignment):
                result = self.backtracking_search(assignment)
                if result is not None:
                    return result
            assignment.pop(first_unassigned)
        
        return None

def main():
    # Define the variables and domains for the Map Coloring problem
    variables = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'Q'],
        'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
        'Q': ['NT', 'SA', 'NSW'],
        'NSW': ['Q', 'SA', 'V'],
        'V': ['SA', 'NSW']
    }
    
    domains = {
        'WA': ['red', 'green', 'blue'],
        'NT': ['red', 'green', 'blue'],
        'SA': ['red', 'green', 'blue'],
        'Q': ['red', 'green', 'blue'],
        'NSW': ['red', 'green', 'blue'],
        'V': ['red', 'green', 'blue']
    }
    
    csp = CSP(variables, domains)
    solution = csp.backtracking_search()
    
    if solution is not None:
        print("Solution found:")
        for var, val in solution.items():
            print(f"{var}: {val}")
    else:
        print("No solution found.")

if __name__ == "__main__":  # Corrected the variable names
    main()       
