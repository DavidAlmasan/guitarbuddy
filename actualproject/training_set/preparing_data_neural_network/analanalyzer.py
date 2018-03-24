import sys, getopt, re

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print("analanalyzer.py -i <inputfile> -o <outputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("analanalyzer.py -i <inputfile> -o <outputfile>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print("Input file is ", inputfile)
    print("Output file is ", outputfile)
    
    #inputfile = "input.txt"
    #outputfile = "output.txt"
    outputfile = open(outputfile, "w")
    with open(inputfile) as f:
        lines = f.readlines()
    lines = [re.split('%s', x) for x in lines] 
    lines = [[int(x[1]), float(x[2])] for x in lines]
    
    data_size = len(lines)
    offset = data_size//10
    data_size = data_size - offset
    prev_line = []
    for i in range(data_size):
        index = lines[i + offset][0]
        accuracy = lines[i + offset][1]
        evolution = accuracy / index * 100
        if(prev_line != []):
            prev_index = prev_line[0]
            prev_accuracy = prev_line[1]
            difference = evolution - (prev_accuracy / prev_index * 100)
            print("{0:.7f}".format(evolution), " ", "{0:.7f}".format(difference), file = outputfile)
        else:
            print("{0:.7f}".format(evolution), file = outputfile)
        prev_line = lines[i + offset]

if __name__ == "__main__":
    main(sys.argv[1: ])
    