from pathlib import Path

main_dir = Path.cwd().joinpath('1_BEM')
data_dir = main_dir.joinpath("data")

if __name__ == "__main__":
    print(main_dir)