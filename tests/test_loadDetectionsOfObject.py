from databaseLoader import loadDetectionsOfObject
import argparse

parser = argparse.ArgumentParser("test loader")
parser.add_argument("db", help="SQLITE DB path")
args = parser.parse_args()

objID = 5
dets = loadDetectionsOfObject(args.db, objID)
print(len(dets))