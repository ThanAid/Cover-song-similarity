"""Download the model h5 file"""
import gdown
import sys

URL = 'https://drive.google.com/uc?id=1sw37Y9V6TBlqgUwZQdjIJbuj9Nf61Nqi'

def main():
    if len(sys.argv) >= 2:
        output = sys.argv[1]
    else:
        output = "DESTINATION_FILE_ON_YOUR_DISK"
    print(f"Downloading model to {output}")
    gdown.download(URL, output, quiet=False)

if __name__ == "__main__":
    main()