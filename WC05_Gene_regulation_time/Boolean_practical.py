from circuitsfunctions import *

########### Part I
# Main function to run the ODE model and Boolean model
def main():
    # ODE model - see the parameters used in circuitsfunctions.py
    ODErun(ODEgeneRegulation, 10, 10)

    # Boolean model
    # print output of function, look at the terminal for the result:
    print("the boolean operation of variableA 1 AND variableB 1 is:", logicalRule(1, 1))  # 1 AND 1 = 1

if __name__ == "__main__":
    main()