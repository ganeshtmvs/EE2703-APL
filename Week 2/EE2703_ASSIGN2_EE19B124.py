"""
        EE2703 Applied Programming Lab - 2019
            Assignment 2 solution
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE2703_ASSIGN2_EE19B124
            commandline INPUT : <FILE NAME> <NETLIST FILE NAME .netlist>
"""

from sys import argv, exit
from numpy import *
CIRCUIT = '.circuit'
END = '.end'
AC = '.ac'
if len(argv) != 2:
    print('\nUsage: %s <inputfile>' % argv[0]) # checking for the correct no.of inputs
    exit()
tokens = {'R':[4],'L':[4],'C':[4],'V':[4,5,6],'I':[4,5,6],'E':[6],'G':[6],'H':[5],'F':[4]}
try:
    with open(argv[1]) as f:
        lines = f.readlines()
        no_of_lines = len(lines)
        start = -3; end = -4;count = 0
        #print(lines, end = "")
        for line in lines:
            if CIRCUIT == line[:len(CIRCUIT)]:
                temp = line
                temp = temp.split('#')[0].split()
                if len(temp[0]) == len(CIRCUIT):
                    count = count + 1
                    start = lines.index(line)
                else:
                    print('Invalid circuit definition')
                    exit(0)
            elif END == line[:len(END)]:
                temp = line
                temp = temp.split('#')[0].split()
                if len(temp[0]) == len(END):
                    #count = count  + 1
                    end = lines.index(line)
                    break
                else:
                    print('Invalid circuit definition')
                    exit(0)
        if start >= end:
            print('Invalid circuit definition')
            #print(count,start,end,temp,len(temp[0]))
            exit(0)
        if count > 1:
             print("THERE ARE MORE COMMANDS OF .circuit THAN REQUIRED")
        circuit_components = []
        list_of_nodes = []
        list_of_sources = []
        w = int(1e-20) #2*pi*frequency
        if len(lines) > end + 1:
            for i in range(end + 1, len(lines)):
                if '.ac' in lines[i]:
                    lines[i] = lines[i].strip().split()
                    w = 2 * pi * float(lines[i][2])
        for line in lines[start+1:end]:
            a = line.split('#')[0].split()
            if len(a) not in tokens[a[0][0]]:
                print("No. of tokens of component {} are incorrect".format(a[0]))
                exit(0)
            if a[0][0] in ('E','G','H','F'):
                print("CIRCUITS HAVING DEPENDENT SOURCES CAN'T BE SOLVED IN THIS CODE")
                exit(0)

            circuit_components.append(a)
            list_of_nodes.append(a[1])
            list_of_nodes.append(a[2])
            if a[0][0] == 'V':
                list_of_sources.append(a[0])
        list_of_nodes = list(set(list_of_nodes))
        list_of_sources = list(set(list_of_sources)) #only Voltage Sources
        matrix_dimension = len(list_of_nodes) + len(list_of_sources)
        A = array([zeros(matrix_dimension) for i in range(matrix_dimension)], dtype=complex)#Conguctance matrix
        b = array([zeros(1) for i in range(matrix_dimension)], dtype=complex)#b matrix
        ref = 0
        ref2 = len(list_of_nodes)
        #Function returns the stamps of particular element
        def matrix_generation(component=[]):
            k = list_of_nodes.index(component[1])#from and to nodes of the element
            l = list_of_nodes.index(component[2])
            if component[0][0] in ('R', 'C', 'L'):
                if component[0][0] == 'R':
                    conductance = 1 / float(component[3]) #conductance of the element
                if component[0][0] == 'C':
                    conductance = complex(0, 1) * ((w * float(component[3])))
                if component[0][0] == 'L':
                    conductance = complex(0, -1 * ((1 / (w * float(component[3])))))
                return conductance, k, l, component[0][0]
            if component[0][0] == 'V':
                if len(component) == 4:
                    Voltage = float(component[3]) #Voltage of source according to given input
                if len(component) == 5:
                    if component[3] == 'ac':
                        Voltage = float(component[4])
                    if component[3] == 'dc':
                        Voltage = float(component[4])
                if len(component) == 6:
                    Voltage = ((float(component[4])) * exp(complex(0, float(component[5]))))
                return Voltage, k, l, component[0][0]
            if component[0][0] == 'I':
                if len(component) == 4:
                    Current = float(component[3]) #Current of source according to given input
                if len(component) == 5:
                    if component[3] == 'ac':
                        Current = float(component[4])
                    if component[3] == 'dc':
                        Current = float(component[4])
                if len(component) == 6:
                    Current = ((float(component[4])) * exp(complex(0, float(component[5]))))
                return Current, k, l, component[0][0]

        #MODIFYING THE MATRICES WITH RESPECTIVE STAMPS
        for i in range(len(circuit_components)):
            result,node_1,node_2,element = matrix_generation(circuit_components[i])
            if element in ('R','C','L'):
                A[node_1,node_1] += result
                A[node_2,node_2] += result
                A[node_1,node_2] -= result
                A[node_2,node_1] -= result
            if element == 'V':
                A[ref2+ref,node_2] = A[node_1,ref2+ref] = 1
                A[ref2 + ref, node_1] = A[node_2, ref2 + ref] = -1
                b[ref2+ref,0] = result
                ref += 1
            if element == 'I':
                b[node_1,0] = result
                b[node_2,0] = -1*result
        x = linalg.solve(A, b) # SOLUTION TO inverse(A)*b
        if 'GND' in list_of_nodes:
            reference_node = 'GND'
            ground = list_of_nodes.index('GND')
            ground_V = x[ground, 0]
        else:
            reference_node = list_of_nodes[0] #if ground is not mentioned and reference is set to a certain node
            ground_V = x[0, 0]
        for i in range(0, len(list_of_nodes)): #MAKING THE VOLTAGES RELATIVE TO REFERENCE NODE
            x[i, 0] -= ground_V
        #print(circuit_components,list_of_sources,list_of_nodes)
        #print(A,b,x)
        #print(w)
        print("The Value of Voltages with refeence to node %s are:"%(reference_node))
        for i in range(0, len(list_of_nodes)):
            print("Voltage at Node %s is %sV" % (list_of_nodes[i], x[i, 0]))
        for i in range(0,len(list_of_sources)):
            print("Current through source %s is %sA" % (list_of_sources[i], x[i + len(list_of_nodes), 0]))
except IOError:
    print('Invalid file')
    exit()

