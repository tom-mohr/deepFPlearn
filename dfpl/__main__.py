import sys

from dfpl.parse import parse_dfpl


def main():
    args = parse_dfpl(*sys.argv[1:])

    # dynamic import after parsing was successful (to allow for faster CLI feedback)
    from dfpl.modes import run_dfpl
    run_dfpl(args)


if __name__ == "__main__":
    main()
