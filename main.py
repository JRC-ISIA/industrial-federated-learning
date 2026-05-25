from config import config
from centralized.run_centralized import run_centralized
from federated.run_federated import run_federated
from hierarchical_federated.run_hierarchical import run_hierarchical


def main():

    paradigm = config["paradigm"]

    if paradigm == "CL":
        run_centralized()

    elif paradigm == "FL":
        run_federated()

    elif paradigm == "H-FL":
        run_hierarchical()

    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")


if __name__ == "__main__":
    main()
