CREATE TABLE IF NOT EXISTS detections (
                                objID INTEGER PRIMARY KEY NOT NULL,
                                frameNum INTEGER NOT NULL,
                                label TEXT NOT NULL,
                                confidence REAL NOT NULL,
                                x REAL NOT NULL,
                                y REAL NOT NULL,
                                width REAL NOT NULL,
                                height REAL NOT NULL
                            );