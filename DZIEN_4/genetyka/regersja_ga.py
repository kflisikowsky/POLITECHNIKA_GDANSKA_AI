import operator, math, random

ops = [
    ("add", operator.add, 2),
    ("mul", operator.mul, 2),
    ("sin", math.sin, 1)
]

def random_tree(depth=3):
    if depth == 0 or random.random() < 0.3:
        return ("var", None)
    op_name, op_fun, arity = random.choice(ops)
    return (op_name, [random_tree(depth-1) for _ in range(arity)])

def eval_tree(tree, x):
    name, children = tree
    if name == "var":
        return x
    op_name, op_fun, arity = [o for o in ops if o[0] == name][0]
    vals = [eval_tree(c, x) for c in children]
    return op_fun(*vals)

def mse(tree):
    return sum((eval_tree(tree, xi) - yi)**2 for xi, yi in zip(X, y))

# Mutacje/crossover pomijam tu dla przejrzystości
