import argparse
import databaseLogger
import argparse

argparser = argparse.ArgumentParser("Testing databaseLogger functions.")
argparser.add_argument("-db", "--database", type=str, help="Path to database file.")
args = argparser.parse_args()

conn = databaseLogger.getConnection(args.database)
print(databaseLogger.getLatestFrame(conn))
    