"""
Script to load detections from database into simple lists.
Befire usage 'export PYTHONPATH="<path to computer_vision_research dir>", 
example 'export PYTHONPATH="$PYTHONPATH:/home/pecneb/gitclones/computer_vision_research/"'
"""
import databaseLoader
import argparse

def getArguments():
    """Get command line arguments.
    Extend this function with arguments.
    Use 'argparser.add_argument(flagname, type, help, ...)

    Returns:
        _type_: _description_
    """
    argparser = argparse.ArgumentParser(prog="Script to fetch data from database in list format.") 
    argparser.add_argument("-db", "--database", type=str, help="The path to the database file.")
    return argparser.parse_args()

def getDetections(path2db: str):
    """Simple dataset loader, that creates lists with the objID and the detections of the object.

    Args:
        path2db (str): Path to database file.

    Returns:
        list: structure of the list [id, detections], 
        where detections is another list [frameNumber, x, y, width, height, vx, vy, ax, ay] 
    """
    retList = []
    # use databaseLoader module to fetch raw data from database
    objects = databaseLoader.loadObjects(path2db)
    # iterate through logged objects
    for obj in objects:
        # load detections of obj
        detections = databaseLoader.loadDetectionsOfObject(path2db, obj[0])
        track = [obj[0], []]
        # iterate through detections of obj to extrack data
        for detection in detections:
            track[1].append([detection[1], detection[3], detection[4], 
                            detection[5], detection[6], detection[7], 
                            detection[8], detection[9], detection[10]])
        retList.append(track)
    return retList

def example_load_and_print_data():
    """This example function shows, how to use getDetections() function, 
       and how to iterate through the returned list.
    """
    dataset = getDetections("research_data/0002_1_37min/0002_1_37min.db")
    for data in dataset:
        print("Object ID: ", data[0])
        for det in data[1]:
            print(f"Framenumber: {det[0]} X: {det[1]} Y: {det[2]}")

def main():
    pass

if __name__ == "__main__":
    main()