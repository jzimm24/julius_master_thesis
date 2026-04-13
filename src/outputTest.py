import time
import sys

def main():
    iterations = 200
    for i in range(iterations):
        print("##################", i, "##################")
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())