# **OMapMaker**
## Webová aplikace / Web app: 
[https://omapmaker.vercel.app/]([url](https://omapmaker.vercel.app/))
## Stažení a spuštění

1) Stáhněte instalační soubor z nejnovějšího release.

2) instalujte

3) Po spuštění aplikace vložte DMR a DMP, nebo vyberte oblast, kterou chcete stáhnout přes tlačítko Download data from ČÚZK. Stažená data se automaticky vloží a stačí spustit generování.
   (Při stahování dat DMP OK přes ATOM někdy dichází k chybě na straně ČÚZK, kterou nejde ovlivnit)

4) Po spuštění "Generate Map" jsou defaultně jsou pak stažena a použita data z OSM.

5) Další nastavení (záložky nahoře):
   - Výška vegetace
   - měřítko (pokud je zvolen extent, použije se 1 : 10 000, které nemusíí odpovídat)
   - Formát papíru
   - Souřadnicový  systém
   - Grivace / magnetická deklinace (odchylka od zvoleného souřadnicového systému
   - Výška kupek, houbka prohlubní, zhlazení vrstevnic
   - možnost vykreslení jen něktarých skupin symbolů
  
### Import mapy do OpenOrienteeringMapperu:
Pokud chcete dostat mapu do do OMM jako jednotlivé vrstvy, je nutné, při generování mapy, zakliknout "Export layers for OOM". Vyexportovaný soubor s příponou .gpkg najdete ve stejné složce jako PNG s mapou. Přes Import v OOM pak stačí nahrát soubor a vybrat CRT soubor OMapMaker-OpenOrienteeringMapper.crt. Mapa by se měla vykreslitsprávnými znaky.
   

### **Data ZABAGED©**
Pokud chcete pracovat s daty ze ZABAGED©, nejjednodušší je si stáhnout celou databázi [tady](https://geoportal.cuzk.cz/(S(p5s5o0ytsichpi2q2qzdlt30))/Default.aspx?mode=TextMeta&text=dSady_zabaged&side=zabaged&menu=24))
Po stažení stačí nahrát do QGIS a exportovat do SHP tímto skriptem:
  
    myDir = 'C:/Vas/Adresar/'
    for vLayer in iface.mapCanvas().layers():
    QgsVectorFileWriter.writeAsVectorFormat(vLayer, myDir + vLayer.name() + ".shp", "utf-8", vLayer.crs(), "ESRI Shapefile")

Po spuštění skriptu stačí nahrát všechny SHP a skript si je přebere. 

<img width="9353" height="7015" alt="Kost" src="https://github.com/user-attachments/assets/7a7d9acb-22f1-4f5b-9b77-84985e5dec14" />
Ukázky vytvořené z OSM dat tady: [https://pkubecek.github.io/MapAnt.cz/]

### **Errors:**
#### **Stahování OSM dat**
1) Pokud se nestahují data z OSM, mělo vby stačit vymazat obsah složky cache,
který se automaticky vytvořil na stejné cestě.

2) Pokud se nestahují data z OSM až do krajů generované oblasti, tak je třeba zvýšit parametry rozsahu dat,
která se stahují.


