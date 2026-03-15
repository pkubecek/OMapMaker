# **OMapMaker**

## Spuštění skriptu

1) Stáhněte nejnovější release a soubory symbols15.xml a symbols10.xml. XML soubory vložte do **stejné složky** jako .ewe soubor.

2) Potom stačí spustit skript a vybrat DMR a DMP, která máte stažená nebo vyberte oblast pomocí tlačítka.

3) Defaultně jsou pak stažena a použita data z OSM.

4) Další nastavení:
   - Výška vegetace
   - měřítko (pokud je zvolen extent, použije se 1 : 10 000, které nemusíí odpovídat)
   - Formát papíru
   - Souřadnicový  systém
   - Grivace / magnetická deklinace (odchylka od zvoleného souřadnicového systému
   - Výška kupek, houbka prohlubní, zhlazení vrstevnic
   - možnost vykreslení jen něktarých skupin symbolů
  
## ** Import mapy do OpenOrienteeringMapperu:**
Pokud chcete dostat mapu do do OMM jako jednotlivé vrstvy, je nutné, při generování mapy, zakliknout "Export layers for OOM". Vyexportovaný soubor s příponou .gpkg najdete ve stejné složce jako PNG s mapou. Přes Import v OOM pak stačí nahrát soubor a vybrat CRT soubor OMapMaker-OpenOrienteeringMapper.crt. Mapa by se měla vykreslitsprávnými znaky.
   

### **Data ZABAGED©**
Pokud chcete pracovat s daty ze ZABAGED©, nejjednodušší je si stáhnout celou databázi [tady](https://geoportal.cuzk.cz/(S(p5s5o0ytsichpi2q2qzdlt30))/Default.aspx?mode=TextMeta&text=dSady_zabaged&side=zabaged&menu=24))
Po stažení stačí nahrát do QGIS a exportovat do SHP tímto skriptem:
  
    myDir = 'C:/Vas/Adresar/'
    for vLayer in iface.mapCanvas().layers():
    QgsVectorFileWriter.writeAsVectorFormat(vLayer, myDir + vLayer.name() + ".shp", "utf-8", vLayer.crs(), "ESRI Shapefile")

Po spuštění skriptu stačí nahrát všechny SHP a skript si je přebere. 

<img width="9353" height="3591" alt="Kost" src="https://github.com/user-attachments/assets/7a7d9acb-22f1-4f5b-9b77-84985e5dec14" />
Ukázky vytvořené z OSM dat tady: [https://pkubecek.github.io/MapAnt.cz/]

### **Errors:**
#### **Stahování OSM dat**
1) Pokud se nestahují data z OSM, mělo vby stačit vymazat obsah složky cache,
který se automaticky vytvořil na stejné cestě.

2) Pokud se nestahují data z OSM až do krajů generované oblasti, tak je třeba zvýšit parametry rozsahu dat,
která se stahují.


