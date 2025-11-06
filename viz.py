from collections import deque

import graphviz


def draw_graph(self):
    dot = graphviz.Graph(comment="Árbol computacional")
    stack = deque([self])

    while stack:
        v = stack.popleft()
        dot.node(str(id(v)), f"{v.label}\ndata={v.data}\ngrad={v.grad}", shape="rectangle")
        if v._op:
            dot.node(str(id(v) + id(v._op)), v._op)
            dot.edge(str(id(v) + id(v._op)), str(id(v)))
        for child in v._prev:
            if v._op:
                dot.edge(str(id(child)), str(id(v) + id(v._op)))
            else:
                dot.edge(str(id(v) + id(v._op)), str(id(v)))

            stack.append(child)

    dot.save('grafo.dot')