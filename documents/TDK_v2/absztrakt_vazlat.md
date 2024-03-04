# Absztrakt vazlat

Az ITS (intelligent transportation system) egyre nagyobb teret
hódít napjainkban és rengeteg különböző területen alkalmazzák
ezeket a rendszereket. A közlekedési csomópontok elemzése
egy frekventált terület az ITS alkalmazásában. Célunk, gépi
látás és gépi tanulás felhasználásával, közlekedési csomópontok
elemzésének automatizálása és felgyorsítása. A kutatásban
lefektetett alapgondolatokat, kifejlesztett keretrendszert és a felmerülő
probélmák megoldásait a gyakorlatban balesetek megelőzésére,
renitens viselkedések kiszűrésére és forgalomirányító renszerek
támogatására lehet használni. A kutatásban egy trajektória osztályozó
módszert ismertetünk, amely objektumdetektálás és objektumkövetés
segítségével elemezi a közlekedési csomópontokban elhaladó járművek
mozgását. A mozgásuk alapján klaszterezi a trajektóriákat.
A klaszterező algoritmus paramétereinek változtatásával különböző
finomságú útvonal szétválasztás érhető el. Ezek az útvonal csoportok
bemenetként szolgálnak az osztályzó algoritmus tanításánál,
ami betanítás után valós időben tudja prediktálni a belépő járművek
kimeneti pontjait. Az OPTICS klaszterező algoritmus bizonyult a mi esetünkben
a leghatásosabbnak. Az OPTICS által csoportosított útvonalak végpontjai alapján
további finomításra van lehetőség egy általunk létrehozott algoritmus segítségével
ami a KMeans algoritmust használja fel alapul. Az útvonalak valós idejű
osztályozására az SVM, KNN és DecisionTree osztályzó algoritmusok érték
el a legmagasabb pontosságot az általunk végzett kiértékelések során.
Továbbá az általunk kifejlesztett keretrendszer alkalmas forgalomszámlálásra
és forgalmi statisztikák előállítására, ami értékes információként szolgálhat
közlekedés mérnököknek.

## TODO

### Ujraklaszterezes

### Forgalomszamlalas heatmapek

#### Klaszterek irányszög szerinti rendezése, be/ki lépési pontok és csak kilépési pontok.

GitHub Copilot: Certainly! Here's a formatted version of the paragraph:

---

# Intelligent Transportation Systems (ITS)

Intelligent Transportation Systems (ITS) are becoming increasingly popular and are being used in a variety of different areas. One area where ITS is frequently used is in the analysis of traffic intersections. Our goal is to automate and speed up the analysis of traffic intersections using computer vision and machine learning. The fundamental ideas, developed framework, and solutions to the problems that arise can be used in practice to prevent accidents, filter out unruly behavior, and support traffic control systems.

In our research, we describe a trajectory classifier method that analyzes the movement of vehicles passing through traffic intersections using object detection and object tracking. Based on their movement, the trajectories are clustered. By changing the parameters of the clustering algorithm, trajectories of different levels of detail can be separated. These groups of trajectories serve as input for the classifier algorithm, which, after training, can predict the exit points of incoming vehicles in real-time.

The OPTICS clustering algorithm proved to be the most effective in our case. Based on the endpoints of the paths grouped by OPTICS, further refinement is possible using an algorithm we created that uses the KMeans algorithm as a basis. The SVM, KNN, and DecisionTree classifier algorithms achieved the highest accuracy in our evaluations of real-time classification of trajectories.

Furthermore, the framework we developed is suitable for traffic counting and generating traffic statistics, which can serve as valuable information for transportation engineers.

# V2

Az ITS (intelligent transportation system) egyre nagyobb teret hódít napjainkban és rengeteg különböző területen alkalmazzák ezeket a rendszereket. A közlekedési csomópontok elemzése egy frekventált terület az ITS alkalmazásokban. Célunk, gépi látás és gépi tanulás felhasználásával, közlekedési csomópontok elemzésének automatizálása és felgyorsítása. A kutatás eredményeit, a kifejlesztett keretrendszert és a felmerülő problémák megoldásait a gyakorlatban balesetek megelőzésére, renitens viselkedések kiszűrésére és forgalomirányító rendszerek támogatására lehet használni. A kutatásban egy trajektória osztályozó módszert ismertetünk, amely objektumdetektálás és objektumkövetés segítségével elemezi a közlekedési csomópontokban elhaladó járművek mozgását. A mozgásuk alapján klaszterezi a trajektóriákat. A klaszterező algoritmus paramétereinek változtatásával különböző finomságú útvonal szétválasztás érhető el. Ezek az útvonal csoportok bemenetként szolgálnak az osztályzó algoritmus tanításánál, ami betanítás után valós időben tudja prediktálni a belépő járművek kimeneti pontjait. Az OPTICS klaszterező algoritmus bizonyult a mi esetünkben a leghatásosabbnak. Az OPTICS által csoportosított útvonalak végpontjai alapján további finomításra van lehetőség egy általunk létrehozott algoritmus segítségével, ami a KMeans algoritmust használja fel alapul. A közlekedők útvonalainak valós idejű előrejelzésére létrehozott módszerünkben az SVM, KNN és DecisionTree osztályzó algoritmusok érték el a legmagasabb pontosságot az általunk végzett kiértékelések során. Továbbá az általunk kifejlesztett keretrendszer alkalmas forgalomszámlálásra és forgalmi statisztikák előállítására, ami értékes információként szolgálhat közlekedésmérnököknek.

# V2 Nyelvtani javitas

Az ITS (intelligent transportation system) egyre nagyobb teret hódít napjainkban és rengeteg különböző területen alkalmazzák ezeket a rendszereket. A közlekedési csomópontok elemzése egy frekventált terület az ITS alkalmazásokban. Célunk, gépi látás és gépi tanulás felhasználásával, közlekedési csomópontok elemzésének automatizálása és felgyorsítása. A kutatás eredményeit, a kifejlesztett keretrendszert és a felmerülő problémák megoldásait a gyakorlatban balesetek megelőzésére, renitens viselkedések kiszűrésére és forgalomirányító rendszerek támogatására lehet használni. A kutatásban egy trajektória osztályozó módszert ismertetünk, amely objektum detektálás és objektumkövetés segítségével elemezze a közlekedési csomópontokban elhaladó járművek mozgását. A mozgásuk alapján klaszterezi a trajektóriákat. A klaszterező algoritmus paramétereinek változtatásával különböző finomságú útvonal szétválasztás érhető el. Ezek az útvonal csoportok bemenetként szolgálnak az osztályozó algoritmus tanításánál, ami betanítás után valós időben tudja prediktálni a belépő járművek kimeneti pontjait. Az OPTICS klaszterező algoritmus bizonyult a mi esetünkben a leghatásosabbnak. Az OPTICS által csoportosított útvonalak végpontjai alapján további finomításra van lehetőség egy általunk létrehozott algoritmus segítségével, ami a KMeans algoritmust használja fel alapul. A közlekedők útvonalainak valós idejű előrejelzésére létrehozott módszerekben az SVM, KNN és DecisionTree osztályozó algoritmusok érték el a legmagasabb pontosságot az általunk végzett kiértékelések során. Továbbá az általunk kifejlesztett keretrendszer alkalmas forgalomszámlálásra és forgalmi statisztikák előállítására, ami értékes információként szolgálhat közlekedésmérnököknek.

# Compress PDF

`qpdf --optimize-images docs/build/latex/trajectorynet.pdf ./PeterBenceGabor_TDK_trajectorynet_documentation.pdf`