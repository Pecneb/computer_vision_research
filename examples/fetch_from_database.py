"""
Script to load detections from database into simple lists.
Befire usage 'export PYTHONPATH="<path to computer_vision_research dir>", example for me 'export PYTHONPATH="/home/pecneb/gitclones/computer_vision_research/"'
"""
import databaseLoader

def getLists(path2db):
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

def main():
    dataset = getLists("research_data/0002_1_37min/0002_1_37min.db")
    for data in dataset:
        print("Object ID: ", data[0])
        for det in data[1]:
            print(f"Framenumber: {det[0]} X: {det[1]} Y: {det[2]}")

if __name__ == "__main__":
    main()