# **OMapMaker**

## Spuštění skriptu

1) Stáhněte nejnovější release a soubory symbols15.xml a symbols10.xml. XML soubory vložte do stejné složky jako EXE soubor.

2) Stáhněte si DMR a DMP zvoleného území

3) Potom stačí spustit skript a vybrat DMR a DMP. Defaultně jsou pak stažena a použita data z OSM.

4) Další nastavení:
   - Výška vegetace
   - měřítko (pokud je zvolen extent, použije se 1 : 10 000, které nemusíí odpovídat)
   - Formát papíru
   - Souřadnicový  systém
   - Grivace / magnetická deklinace (odchylka od zvoleného souřadnicového systému
   - Výška kupek, houbka prohlubní, zhlazení vrstevnic
   - možnost vykreslení jen něktarých skupin symbolů
   

### **Data ZABAGED©**
Pokud chcete pracovat s daty ze ZABAGED©, nejjednodušší je si stáhnout celou databázi [tady](https://geoportal.cuzk.cz/(S(p5s5o0ytsichpi2q2qzdlt30))/Default.aspx?mode=TextMeta&text=dSady_zabaged&side=zabaged&menu=24))
Po stažení stačí nahrát do QGIS a exportovat do SHP tímto skriptem:
  
    myDir = 'C:/Vas/Adresar/'
    for vLayer in iface.mapCanvas().layers():
    QgsVectorFileWriter.writeAsVectorFormat(vLayer, myDir + vLayer.name() + ".shp", "utf-8", vLayer.crs(), "ESRI Shapefile")

Po spuštění skriptu stačí nahrát všechny SHP a skript si je přebere. 

<img width="9353" height="3591" alt="NACH24DMR_OMap" src="https://github.com/user-attachments/assets/97a1db28-f29c-4e41-838f-ccca1ac6ec75" />
Ukázky vytvořené z OSM dat tady: [https://pkubecek.github.io/MapAnt.cz/]

### **Errors:**
#### **Stahování OSM dat**
1) Pokud se nestahují data z OSM, mělo vby stačit vymazat obsah složky cache,
který se automaticky vytvořil na stejné cestě.

2) Pokud se nestahují data z OSM až do krajů generované oblasti, tak je třeba zvýšit parametry rozsahu dat,
která se stahují.


