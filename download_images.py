from google_images_download import google_images_download   #importing the library
import argparse

parser = argparse.ArgumentParser(description="Google crawler")
parser.add_argument('-o', '--output_directory', type=str, default="",\
     help="""Allows you specify the main directory name in which the images are downloaded.
     If not specified, it will default to ‘downloads’ directory. This directory is located in the path from where you run this code
     The directory structure would look like: <output_directory><image_directory><images>""")
parser.add_argument('-k', '--keywords', type=str,\
     help="Denotes the keywords/key phrases you want to search for. For more than one keywords, wrap it in single quotes.")
parser.add_argument('-sk', '--suffix_keywords', type=str,\
     default="",\
     help="Denotes additional words added after main keyword while making the search query. The final search query would be: <keyword> <suffix keyword>")
parser.add_argument('-l', '--limit', type=int, default=100,\
     help="Denotes number of images that you want to download.")
parser.add_argument('-cd', '--chromedriver', type=str, default="",\
     help="In case you want to download larger than 100 images with this argument you can pass the path to the ‘chromedriver’.")
args = parser.parse_args()

def main(args):
    print(args)
    response = google_images_download.googleimagesdownload()   #class instantiation
    arguments = {"keywords":args.keywords, "suffix_keywords":args.suffix_keywords,\
         "limit":args.limit, "print_urls":True, "output_directory":args.output_directory,\
         "chromedriver":args.chromedriver}   #creating list of arguments
    paths = response.download(arguments)   #passing the arguments to the function
    print(paths)   #printing absolute paths of the downloaded images

if __name__ == "__main__":
    main(args)