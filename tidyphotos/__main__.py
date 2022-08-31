import shlex
import sys
import os

def print_err(phrase: str) -> None:
    print("ERROR: " + phrase)

def main() -> int:
    if len(sys.argv) != 3:
        print_err("This app needs exactly 2 arguments: SOURCE_DIR and OUTPUT_DIR.")
        return 1
    
    src_dir = os.path.abspath(sys.argv[1])
    dest_dir = os.path.abspath(sys.argv[2])
    
    if not os.path.isdir(src_dir) or len(src_dir) == 0:
        print_err(f"SOURCE_DIR ERROR: '{src_dir}' is not a valid existing directory.")
        return 1
    
    if os.path.isdir(dest_dir):
        print_err(f"OUTPUT_DIR ERROR: '{dest_dir}' already exists and cannot be used as output directory.")
        return 1
    
    phrase = shlex.join(sys.argv[1:])
    print("SOURCE_DIR: " + src_dir)
    print("OUTPUT_DIR: " + dest_dir)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
    
