from misra_aug import augment
import os


def main():

    filepath = "../preprocessing/data_control_preprocessed.csv"
    save_path1 = "../classifiers/control_aug.csv"
    save_path2 = "../classifiers/canonical_control_aug.csv"

    augment(filepath, save_path1)
    augment(filepath, save_path2, include_canonical=True)


if __name__ == "__main__":
    main()
