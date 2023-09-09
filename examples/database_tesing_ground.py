"""
    Predicting trajectories of objects
    Copyright (C) 2022  Bence Peter

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Contact email: ecneb2000@gmail.com
"""
import argparse
import utils.databaseLogger
import utils.databaseLoader as databaseLoader

argparser = argparse.ArgumentParser("Testing databaseLogger functions.")
argparser.add_argument("-db", "--database", type=str, help="Path to database file.")
args = argparser.parse_args()

detections = databaseLoader.loadDetectionsOfObject(args.database, 1)
for det in detections:
    print(det)