"""
        EE2703 Applied Programming Lab - 2019
            Assignment 1 solution
            ROLL No. EE19B124
            Name : T.M.V.S GANESH
            FILE NAME : EE19B124_file1
            commandline INPUT : <FILE NAME> <NETLIST FILE NAME .netlist> 
"""

from sys import argv, exit

CIRCUIT = '.circuit'
END = '.end'     #Variables for the commands in netlist file


if len(argv) != 2:
    print('\nUsage: %s <inputfile>' % argv[0]) # checking for the correct no.of inputs
    exit()
tokens = {'R':4,'L':4,'C':4,'V':4,'I':4,'E':6,'G':6,'H':5,'F':4} #reference dictonary for no.of tokens
""" 
    try except to check if the filename entered is correct
"""    
try:
    with open(argv[1]) as f:
        lines = f.readlines()
        start = -3; end = -4;count = 0
        #print(lines, end = "")
        for line in lines:              # extracting circuit definition start and end lines
            if CIRCUIT == line[:len(CIRCUIT)]:
                """
                CIRCUIT == line[:len(CIRCUIT)] doesn't take care whether comand is .circuit or .circuit...(Ex:.circuit2 or .endpython)
                """
                temp = line  #temporary variable to store line since line is used to get index
                temp = temp.split('#')[0].split() #list containing the command    ##there can be comments in this line
                if len(temp[0]) == len(CIRCUIT): # to check whether is correct or not 
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
        if start >= end:                # validating circuit block
            print('Invalid circuit definition')
            #print(count,start,end,temp,len(temp[0]))
            exit(0)
        circuit_components = [] #list of lists containing the information of components for a valid netlist
        for line in lines[start+1:end]:
            a = line.split('#')[0].split()
            if tokens[a[0][0]] != len(a):  
                """
                It checks whether the no of tokens for every component matches with that of reference
                """
                print("No. of tokens of component {} are incorrect".format(a[0]))
                exit(0)
            circuit_components.append(a)
            #print(a)

        for line in reversed([' '.join(reversed(line.split('#')[0].split())) for line in lines[start+1:end]]):
            """
        It takes care of
                1. Removing comments
                2. Splitting the words
                3. Reversing words
                4. Reversing lines
            """
            print(line)                 # print output
        #print(count,start,end,temp)
        #print(circuit_components)
        if count > 1:
             print("THERE ARE MORE COMMANDS OF .circuit THAN REQUIRED")
except IOError:
    print('Invalid file')
    exit()
"""
    Along with the errors taken care in the sample solution,The above program can handle some more errors and 
    store the circuit information.
    Additonal errors taken care of :
        1.whether the commands in the file are same as that of reference one
            ex : .circuit and .circuitpython are different but line[:len(CIRCUIT)] of both are equal
        2.whether the no.of tokens of component are equal with that of reference value 
            ex:[ R4 2 in3 8e3 2 ]is incorrect since resistor should have only 4 tokens
        3.A warning message if .circuit appears more than once in netlist
            ex: .circuit #command
                .circuit            

"""


