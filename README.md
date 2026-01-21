# **OMapMaker**

## Spuštění skriptu

1) Stáhněte si OMapMaker.py a xml soubory (symbols15.xml, symbols10.xml)

2) Stáhněte si DMR a DMP zvoleného území

3) Potom stačí spustit skript a vybrat DMR a DMP. Defaultně jsou pak stažena a použita data z OSM.

4) Zatím je možné ladit jen vášku vykreslované vegetace a vybrat měřítko (pokud je vybrán extent, použije se 1 : 10 000 a velikosti symbolů nesedí).

### Data ZABAGED©
Pokud chcete pracovat s daty ze ZABAGED©, nejjednodušší je si stáhnout celou databázi [tady]([url](https://geoportal.cuzk.cz/(S(p5s5o0ytsichpi2q2qzdlt30))/Default.aspx?mode=TextMeta&text=dSady_zabaged&side=zabaged&menu=24))
Po stažení stačí nahrát do QGIS a exportovat do SHP tímto skriptem:
  
    myDir = 'C:/Vas/Adresar/'
    for vLayer in iface.mapCanvas().layers():
    QgsVectorFileWriter.writeAsVectorFormat(vLayer, myDir + vLayer.name() + ".shp", "utf-8", vLayer.crs(), "ESRI Shapefile")

Po spuštění skriptu stačí nahrát všechny SHP a skript si je přebere. 
### **Errors:**
#### **Stahování OSM dat**
1) Pokud se nestahují data z OSM, mělo vby stačit vymazat obsah složky cache,
který se automaticky vytvořil na stejné cestě.

2) Pokud se nestahují data z OSM až do krajů generované oblasti, tak je třeba zvýšit parametry rozsahu dat,
která se stahují.


