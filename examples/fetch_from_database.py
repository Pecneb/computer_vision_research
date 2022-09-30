"""
Script to load detections from database into simple lists.
Befire usage 'export PYTHONPATH="<path to computer_vision_research dir>", 
example 'export PYTHONPATH="$PYTHONPATH:/home/pecneb/gitclones/computer_vision_research/"'
"""
import databaseLoader
from dataAnalyzer import preprocess_database_data_multiprocessed
import argparse
import tqdm

def getArguments():
    """Get command line arguments.
    Extend this function with arguments.
    Use 'argparser.add_argument(flagname, type, help, ...)

    Returns:
        args: Command line arguments class 
    """
    argparser = argparse.ArgumentParser(prog="Script to fetch data from database in list format.") 
    argparser.add_argument("-db", "--database", type=str, help="The path to the database file.")
    argparser.add_argument("--txtfile", type=str, help="Path to txt file.")
    argparser.add_argument("--n_jobs", type=int, help="Processes to run.")
    argparser.add_argument("--txt2dets", help="Parse txt to detections.", action="store_true", default=False)
    argparser.add_argument("--db2txt", help="Parse db data to txt.", action="store_true", default=False)
    return argparser.parse_args()

def getDetections(path2db: str):
    """Simple dataset loader, that creates lists with the objID and the detections of the object.

    Args:
        path2db (str): Path to database file.

    Returns:
        list: structure of the list [id, detections], 
        where detections is another list [frameNumber, x, y] 
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
            track[1].append([detection[1], detection[3], detection[4]])
        retList.append(track)
    return retList

def parseDetsToTxt(path2db: str, n_jobs: int):
    """Convert db data to txt data.

    Args:
        path2db (str): Path to database file. 
        n_jobs (int): Process number. 
    """
    with open(f"detections_{path2db.split('/')[-1].split('.')[0]}.txt", 'w') as out:
        out.write("id, detections(frameNumber, x, y)...;\n")
        objects = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
        for obj in tqdm.tqdm(objects, desc="Iterating through objects"):
            if len(obj.history) > 0:
                towrite = f"{obj.objID}"
                for det in obj.history:
                    towrite += f",{det.frameID}:{det.X:.4f}:{det.Y:.4f}"
                towrite += ';\n'
                out.write(towrite)

def parseTxtToDets(path2txt: str):
    tracks = []
    with open(path2txt) as f:
        header = f.readline()
        while(True):
            txtObj = f.readline().split(',')
            if txtObj[0] == '':
                break
            tracks.append([int(txtObj[0]), []])
            for i in range(1, len(txtObj)):
                det = txtObj[i].split(':')
                if det[2][-2:] == ';\n':
                    tracks[-1][1].append([int(det[0]), float(det[1]), float(det[2][:-2])])
                else:
                    tracks[-1][1].append([int(det[0]), float(det[1]), float(det[2])])
    return tracks
                
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
    args = getArguments()
    if args.db2txt:
        parseDetsToTxt(args.database, args.n_jobs)
    if args.txt2dets:
        parseTxtToDets(args.txtfile)

if __name__ == "__main__":
    main()