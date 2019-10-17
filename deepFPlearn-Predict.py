import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# import my own functions for deepFPlearn
import dfplmodule as dfpl

# ------------------------------------------------------------------------------------- #

def parseInput():
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    parser = argparse.ArgumentParser(description='Use models that have been generated by deepFPlearn-Train.py '
                                                 'tool to make predictions on chemicals (provide SMILES or '
                                                 'topological fingerprints).')
    parser.add_argument('-i', metavar='FILE', type=str, nargs=1,
                        help="The file containin the data for the prediction. It is in"
                             "comma separated CSV format. The column named 'smiles' or 'fp'"
                             "contains the field to be predicted. Please adjust the type "
                             "that should be predicted (fp or smile) with -t option appropriately."
                             "An optional column 'id' is used to assign the outcomes to the"
                             "original identifieres. If this column is missing, the results are"
                             "numbered in the order of their appearance in the input file."
                             "A header is expected and respective column names are used.",
                        required=True)
    parser.add_argument('-m', metavar='FILE', type=str, nargs=1,
                        help='The file of the trained model that should be used for the prediction',
                        required=True)
    parser.add_argument('-o', metavar='FILE', type=str, nargs=1,
                        help='Output file name. It containes a comma separated list of '
                             "predictions for each input row, for all targets. If the file 'id'"
                             "was given in the input, respective IDs are used, otherwise the"
                             "rows of output are numbered and provided in the order of occurence"
                             "in the input file.")
    parser.add_argument('-t', metavar='STR', type=str, nargs=1, choices=['fp', 'smiles'],
                        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
                        required=True)
    parser.add_argument('-k', metavar='STR', type=str, nargs=1,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default=['topological'])

    return parser.parse_args()

# ------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------- #



# ------------------------------------------------------------------------------------- #



# ------------------------------------------------------------------------------------- #


# ===================================================================================== #


if __name__ == '__main__':

    # get all arguments
    args = parseInput()

    #print(args)
    #exit(0)

    # transform X to feature matrix
    # -i /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/multiAOPtox.smiles.csv
    # -m /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/model.AR.h5
    # -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/prediction/multiAOPtox.smiles.predictions.AR.csv
    # -t smile -k topological
    #xpd = dfpl.XfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/multiAOPtox.smiles.csv", rtype="smile", fptype="topological", printfp=True)

    xpd = dfpl.XfromInput(csvfilename=args.i[0], rtype=args.t[0], fptype=args.k[0], printfp=True)

    # predict values for provided data and model
    #ypredictions = dfpl.predictValues(modelfilepath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/2019-10-16_311681247_1000/model.Aromatase.h5", pdx=xpd)
    ypredictions = dfpl.predictValues(modelfilepath=args.m[0], pdx=xpd)


    # write predictions to usr provided .csv file
    pd.DataFrame.to_csv(ypredictions, args.o[0])
