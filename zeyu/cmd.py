import subprocess


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, universal_newlines=True, shell=True).strip(
            "\n"
        )
    except subprocess.CalledProcessError as e:
        print("WARNING: " + str(e))
        return None
