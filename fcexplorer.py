#!/usr/bin/env python3

import itertools
import os
import subprocess
import sys

def generate_combinations(options, option_values):
    # Generate all possible combinations of values for the options
    value_combinations = []
    for opt in options:
        if option_values[opt] == []:
            value_combinations.append([None, ''])
        else:
            value_combinations.append(option_values[opt])
    return itertools.product(*value_combinations)

def main(option_values, files):
    listopt = list(option_values.keys())

    for file in files:
        if not file.endswith('.dsp'):
            continue

        filename_no_ext = os.path.splitext(os.path.basename(file))[0]

        for combination in generate_combinations(listopt, option_values):
            cmd = ['faust']
            output_suffix = []

            for opt, val in zip(listopt, combination):
                if val is None:
                    cmd.append(opt)
                    output_suffix.append(opt[1:])
                elif val != '':
                    cmd.extend([opt, str(val)])
                    output_suffix.append(f"{opt[1:]}_{val}")

            output_file = f"{filename_no_ext}_{'_'.join(output_suffix)}.cpp"
            cmd.extend([file, '-o', output_file])

            print(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, text=True)

if __name__ == "__main__":
    args = sys.argv[1:]

    option_values = {}
    current_option = None
    files = []

    for arg in args:
        if arg.startswith('-'):
            current_option = arg
            option_values[current_option] = []
        elif arg.endswith('.dsp'):
            files.append(arg)
        else:
            if current_option:
                option_values[current_option] = arg.split()
                current_option = None

    main(option_values, files)