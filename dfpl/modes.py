import dataclasses
import logging
import os
import pathlib
from argparse import Namespace
from os import path

import chemprop
from keras.models import load_model

from dfpl import autoencoder as ac
from dfpl import feedforwardNN as fNN
from dfpl import fingerprint as fp
from dfpl import options
from dfpl import single_label_model as sl
from dfpl import vae as vae
from dfpl.parse import parse_dfpl
from dfpl.predictions import predict_values
from dfpl.utils import createDirectory, makePathAbsolute


def run_dfpl(args: Namespace):
    subprogram_name = args.method
    del args.method  # the subprograms don't expect the ".method" attribute
    {
        "traingnn": traindmpnn,
        "predictgnn": predictdmpnn,
        "interpretgnn": interpretdmpnn,
        "train": train,
        "predict": predict,
        "convert": convert
    }[subprogram_name](args)


def traindmpnn(args: Namespace) -> None:
    """
    Train a D-MPNN model using the given options.
    """
    createLogger("traingnn.log")
    opts = options.GnnOptions(**vars(args))
    logging.info("Training DMPNN...")
    mean_score, std_score = chemprop.train.cross_validate(
        args=opts, train_func=chemprop.train.run_training
    )
    logging.info(f"Results: {mean_score:.5f} +/- {std_score:.5f}")


def predictdmpnn(args: Namespace) -> None:
    """
    Predict the values using a trained D-MPNN model with the given options.
    """
    createLogger("predictgnn.log")
    opts = options.PredictGnnOptions(**vars(args))
    chemprop.train.make_predictions(args=opts)


def interpretdmpnn(args: Namespace) -> None:
    """
    Interpret the predictions of a trained D-MPNN model with the given options.
    """
    createLogger("interpretgnn.log")
    opts = options.InterpretGNNoptions(**vars(args))
    chemprop.interpret.interpret(args=opts, save_to_csv=True)


def train(args: Namespace):
    """
    Run the main training procedure
    """
    train_opts = options.TrainOptions(**vars(args))
    opts = dataclasses.replace(
        train_opts,
        inputFile=makePathAbsolute(train_opts.inputFile),
        outputDir=makePathAbsolute(train_opts.outputDir),
    )
    createDirectory(opts.outputDir)
    createLogger(path.join(opts.outputDir, "train.log"))
    logging.info(
        f"The following arguments are received or filled with default values:\n{opts}"
    )
    # import data from file and create DataFrame
    if "tsv" in opts.inputFile:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importDstoxTSV, fp_size=opts.fpSize
        )
    else:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
        )
    # initialize (auto)encoders to None
    encoder = None
    autoencoder = None
    if opts.trainAC:
        if opts.aeType == "deterministic":
            encoder, train_indices, test_indices = ac.train_full_ac(df, opts)
        elif opts.aeType == "variational":
            encoder, train_indices, test_indices = vae.train_full_vae(df, opts)
        else:
            raise ValueError(f"Unknown autoencoder type: {opts.aeType}")

    # if feature compression is enabled
    if opts.compressFeatures:
        if not opts.trainAC:
            # load default options for autoencoder from config file
            compression_options = load_compression_options()
            if opts.aeType == "variational":
                (autoencoder, encoder) = vae.define_vae_model(opts=compression_options)
            else:
                (autoencoder, encoder) = ac.define_ac_model(opts=compression_options)

            if opts.ecWeightsFile == "":
                encoder = load_model(opts.ecModelDir)
            else:
                autoencoder.load_weights(
                    os.path.join(opts.ecModelDir, opts.ecWeightsFile)
                )
        # compress the fingerprints using the autoencoder
        df = ac.compress_fingerprints(df, encoder)
        if opts.visualizeLatent and opts.trainAC:
            ac.visualize_fingerprints(
                df,
                train_indices=train_indices,
                test_indices=test_indices,
                save_as=f"{opts.ecModelDir}/UMAP_{opts.aeSplitType}.png",
            )
        elif opts.visualizeLatent:
            logging.info(
                "Visualizing latent space is only available if you train the autoencoder. Skipping visualization."
            )

    # train single label models if requested
    if opts.trainFNN and not opts.enableMultiLabel:
        sl.train_single_label_models(df=df, opts=opts)

    # train multi-label models if requested
    if opts.trainFNN and opts.enableMultiLabel:
        fNN.train_nn_models_multi(df=df, opts=opts)


def predict(args: Namespace) -> None:
    """
    Run prediction given specific options
    """

    predict_opts = options.PredictOptions(**vars(args))
    opts = dataclasses.replace(
        predict_opts,
        inputFile=makePathAbsolute(predict_opts.inputFile),
        outputDir=makePathAbsolute(predict_opts.outputDir),
        outputFile=makePathAbsolute(
            path.join(predict_opts.outputDir, predict_opts.outputFile)
        ),
        ecModelDir=makePathAbsolute(predict_opts.ecModelDir),
        fnnModelDir=makePathAbsolute(predict_opts.fnnModelDir),
    )
    createDirectory(opts.outputDir)
    createLogger(path.join(opts.outputDir, "predict.log"))
    logging.info(
        f"The following arguments are received or filled with default values:\n{args}"
    )

    # import data from file and create DataFrame
    if "tsv" in opts.inputFile:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importDstoxTSV, fp_size=opts.fpSize
        )
    else:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
        )

    if opts.compressFeatures:
        # load trained model for autoencoder
        compression_options = load_compression_options()
        if opts.aeType == "deterministic":
            (autoencoder, encoder) = ac.define_ac_model(opts=compression_options)
        if opts.aeType == "variational":
            (autoencoder, encoder) = vae.define_vae_model(opts=compression_options)
        # Load trained model for autoencoder
        if opts.ecWeightsFile == "":
            encoder = load_model(opts.ecModelDir)
        else:
            encoder.load_weights(os.path.join(opts.ecModelDir, opts.ecWeightsFile))
        df = ac.compress_fingerprints(df, encoder)

    # Run predictions on the compressed fingerprints and store the results in a dataframe
    df2 = predict_values(df=df, opts=opts)

    # Extract the column names from the dataframe, excluding the 'fp' and 'fpcompressed' columns
    names_columns = [c for c in df2.columns if c not in ["fp", "fpcompressed"]]

    # Save the predicted values to a CSV file in the output directory
    df2[names_columns].to_csv(path_or_buf=path.join(opts.outputDir, opts.outputFile))

    # Log successful completion of prediction and the file path where the results were saved
    logging.info(
        f"Prediction successful. Results written to '{path.join(opts.outputDir, opts.outputFile)}'"
    )


def load_compression_options() -> options.TrainOptions:
    project_directory = pathlib.Path(__file__).parent.absolute()
    args = parse_dfpl("train",
                      configFile=makePathAbsolute(f"{project_directory}/compression.json"))
    return options.TrainOptions(**vars(args))


def convert(args: Namespace):
    directory = makePathAbsolute(args.f)
    if path.isdir(directory):
        createLogger(path.join(directory, "convert.log"))
        logging.info(f"Convert all data files in {directory}")
        fp.convert_all(directory)
    else:
        raise ValueError("Input directory is not a directory")


def createLogger(filename: str) -> None:
    """
    Set up a logger for the main function that also saves to a log file
    """
    # get root logger and set its level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(filename, mode="w")
    fh.setLevel(logging.INFO)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatterFile = logging.Formatter(
        "{asctime} - {name} - {levelname} - {message}", style="{"
    )
    formatterConsole = logging.Formatter("{levelname} {message}", style="{")
    fh.setFormatter(formatterFile)
    ch.setFormatter(formatterConsole)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
