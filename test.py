from engine import Value

alfa = Value(10, label="ALFA")
beta = Value(-2, label="BETA")
gamma = Value(5, label="GAMMA")

a = 10
b = -7
c = 4

alfabeta = alfa * beta; alfabeta.label = "ALFA x BETA"
alfagamma = alfa * gamma; alfagamma.label = "ALFA x GAMMA"
suma = alfabeta + alfagamma; suma.label = "ALFABETA + ALFAGAMMA"
mifun = suma / beta; mifun.label = "mifun"
mifun.backward()
mifun.draw_graph()