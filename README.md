# **OMapMaker**
### **Errors:**
#### **OSM data**
--> Pokud se nestahují data z OSM, mělo vby stačit vymazat obsah složky cache,
který se automaticky vytvořil na stejné cestě.
--> Pokud se nestahují data z OSM až do krajů generované oblasti, tak je třeba zvýšit parametry rozsahu dat,
která se stahují.
### **Soubor ZABAGED_to_ISOM_2017-2:**
obsahuje názvy dat vrstev v ZABAGED a jejich odpovídající názvy, které je možné využát jako
vstup do OMapMaker. Názvy .shp souborů jsou schodné jako v ISOM 2017-2. 
Použijte CamelCase např. WaterCourse.shp, NarrowRide.shp, VehicleTrack.shp
